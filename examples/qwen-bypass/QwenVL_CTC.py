import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import Optional, Union

from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures
from transformers.vision_utils import get_vision_bilinear_indices_and_weights, get_vision_position_ids


class Qwen3VLWrapper:
    """
    Class to extract and put back intermediate features from a Qwen3VL model.  
    Serves as standard API in the CoFAI framework.  
    Currently only supports breakpoint on 1st deepstack index (mandatory breakpoint).  
        
    Available methods:  
    - prepare_image: Preprocess a PIL image (H,W) and return pixel_values and image_grid_thw. Expand/shrink to min/max pixels.  
    - extract_features: Run the visual encoder up to the first deepstack index (slot 9) and return intermediate hidden state (1152,H/16,W/16) from that slot.  
    - generate_from_features: Continue running the visual encoder to compute all features (including deepstack features), and then run the LM to generate text.  
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        default_prompt: str = "",
        local_files_only: bool = True,
        dtype: str = "auto",
        min_pixels: Optional[int] = 768*28*28,
        max_pixels: Optional[int] = 1536*28*28
    ):
        self.model_path = model_path
        self.device = device
        self.default_prompt = default_prompt
        self.local_files_only = local_files_only
        self.dtype = dtype

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device,
            local_files_only=local_files_only,
            attn_implementation="eager"
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.hidden_size = self.model.config.text_config.hidden_size # 1152
        self.spatial_merge_size = self.model.config.vision_config.spatial_merge_size
        self.image_token_id = self.model.config.image_token_id

        # Find 1st deepstack index (=Breakpoint)
        self.first_ds_idx = self.model.model.visual.deepstack_visual_indexes[0] # After Layer 8, at slot 9
        

        print(" -- Qwen3VL is initialized -- ")
        print(f"  - Device: {device}")
        print(f"  - Breakpoint: Layer {self.first_ds_idx}")

    # Internal utility functions:
    def _get_prompt_template(self, prompt: str):
        '''
        After codec, original input feature is not available.  
        This utility function generates a dummy input for the LM, allowing re-injection of feature_hat later.
        ''' 
        dummy_img = Image.new("RGB", (64, 64))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dummy_img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        dummy_input = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        input_ids = dummy_input.input_ids[0]
        mm_types = dummy_input.mm_token_type_ids[0] if hasattr(dummy_input, 'mm_token_type_ids') \
                   else torch.zeros_like(input_ids)

        img_positions = (input_ids == self.image_token_id).nonzero(as_tuple=True)[0]
        assert len(img_positions) > 0

        img_start = img_positions[0].item()
        img_end = img_positions[-1].item() + 1

        template = {
            'prefix_ids': input_ids[:img_start],
            'suffix_ids': input_ids[img_end:],
            'prefix_mm': mm_types[:img_start],
            'suffix_mm': mm_types[img_end:],
        }
        return template

    def _build_inputs(self, prompt: str, num_visual_tokens: int):
        '''
        Passes input to LM, with dummy visual input.
        '''
        template = self._get_prompt_template(prompt)
        device = self.device

        prefix_ids = template['prefix_ids'].to(device)
        suffix_ids = template['suffix_ids'].to(device)
        img_ids = torch.full((num_visual_tokens,), self.image_token_id, dtype=torch.long, device=device)
        input_ids = torch.cat([prefix_ids, img_ids, suffix_ids]).unsqueeze(0)

        prefix_mm = template['prefix_mm'].to(device)
        suffix_mm = template['suffix_mm'].to(device)
        img_mm = torch.ones(num_visual_tokens, dtype=torch.int, device=device)
        mm_token_type_ids = torch.cat([prefix_mm, img_mm, suffix_mm]).unsqueeze(0)

        attention_mask = torch.ones_like(input_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'mm_token_type_ids': mm_token_type_ids,
            'prompt_len': input_ids.shape[1],
        }

    def prepare_image(self, image: Union[str, Image.Image]):
        '''
        Prepare a PIL image into pixel_values and image_grid_thw for the model.
        '''
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": 
                [
                    {"type": "image","image": image},
                    {"type": "text", "text": "dummy"}
                ],
            }
        ]
            
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        
        return pixel_values, image_grid_thw

    def extract_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> dict:
        '''
        Runs the Visual Encoder up to the first deepstack index, and returns hidden states.
        ''' 
        visual = self.model.model.visual
        device = self.device
        
        block_dtype = next(visual.blocks[0].parameters()).dtype
        
        hidden_states = visual.patch_embed(pixel_values.to(block_dtype).to(device))
        
        # make PE
        bilinear_indices, bilinear_weights = get_vision_bilinear_indices_and_weights(
            image_grid_thw.to(device),
            num_grid_per_side=visual.num_grid_per_side,
            spatial_merge_size=visual.spatial_merge_size,
        )
        pos_embeds = (visual.pos_embed(bilinear_indices) * bilinear_weights[:, :, None]).sum(0)
        
        # apply PE
        hidden_states = hidden_states + pos_embeds.to(device).to(block_dtype)

        position_ids = get_vision_position_ids(image_grid_thw.to(device), visual.spatial_merge_size)
        rotary_pos_emb = visual.rotary_pos_emb(position_ids)
        
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)

        cos_emb = emb.cos()
        sin_emb = emb.sin()
        position_embeddings = (cos_emb, sin_emb)

        cu_seqlens = torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2],
            image_grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(device)

        # run until 1st deepstack index
        for layer_num, blk in enumerate(visual.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num == self.first_ds_idx:
                break
                
        h = image_grid_thw[0, 1].item() 
        w = image_grid_thw[0, 2].item() 
        
        # NOTE: for researchers' convenience, hidden_states is given in its 2D spatial format (1152, H, W), instead of flat (H*W, 1152)
        hidden_states = hidden_states.reshape(h, w, -1).permute(2, 0, 1)
                                
        return hidden_states

    def generate_from_features(
        self,
        hidden_states: torch.Tensor,  # [1152, h, w]
        prompt: Optional[str] = "",
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 1.0,
        what_to_return: str = "string", # string or ids
        **gen_kwargs
    ):
        '''
        Receives reconstructed hidden_states [1152, h, w], continues visual encoder,
        passes all features+deepstack to LM, gets sentence.
        '''
        visual = self.model.model.visual
        device = self.device
        
        hidden_size, h, w = hidden_states.shape
        
        # reconstruct image_grid_thw: [[1, h, w]]
        image_grid_thw = torch.tensor([[1, h, w]], device=device, dtype=torch.int64)
        
        # reconstruct cu_seqlens
        cu_seqlens = torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2],
            image_grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(device)
        
        # reconstruct position_embeddings
        
        position_ids = get_vision_position_ids(image_grid_thw.to(device), visual.spatial_merge_size)
        rotary_pos_emb = visual.rotary_pos_emb(position_ids)
        
        seq_len = h * w
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        position_embeddings = (cos_emb, sin_emb)
        
        # flatten to sequential
        hidden_states = hidden_states.to(visual.dtype).to(device)
        hidden_states = hidden_states.permute(1, 2, 0).reshape(h * w, hidden_size)

        # Run the rest of visual blocks, and collect deepstack features
        deepstack_features = []

        for layer_num, blk in enumerate(visual.blocks[self.first_ds_idx:], start=self.first_ds_idx):
            if layer_num != self.first_ds_idx:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                )
            if layer_num in visual.deepstack_visual_indexes:
                idx = visual.deepstack_visual_indexes.index(layer_num)
                ds_feat = visual.deepstack_merger_list[idx](hidden_states)
                deepstack_features.append(ds_feat) # 1st input to LM: deepstack visual features

        pooler_output = visual.merger(hidden_states) # 2nd input to LM: pooled visual features
        num_visual_tokens = pooler_output.shape[0]

        inputs = self._build_inputs(prompt, num_visual_tokens)

        # CORE: create a monkey patch forward function for LM
        # which uses the above reconstructed visual features
        original_visual_forward = visual.forward

        def patched_visual_forward(_hidden_states, grid_thw, **kwargs):
            return BaseModelOutputWithDeepstackFeatures(
                last_hidden_state=hidden_states,
                pooler_output=pooler_output,
                deepstack_features=deepstack_features,
            )

        visual.forward = patched_visual_forward
        
        dummy_pixel_values = torch.zeros(1, 3, 224, 224, device=self.device, dtype=visual.dtype)
        try:
            generation_inputs = {
                "input_ids": inputs['input_ids'],
                "attention_mask": inputs['attention_mask'],
                "mm_token_type_ids": inputs['mm_token_type_ids'],
                "pixel_values": dummy_pixel_values, # Must pass a dummy. Otherwise the LM will believe task is text-only, and will ignore visual.
                "image_grid_thw": image_grid_thw,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "use_cache": True,
                "pad_token_id": self.processor.tokenizer.pad_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
                **gen_kwargs
            }

            with torch.no_grad():
                # run LM generate as usual
                generated_ids = self.model.generate(**generation_inputs)
                
                new_ids = generated_ids[:, inputs['input_ids'].shape[1]:]

                output_text = self.processor.batch_decode(
                    new_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

            if what_to_return == "ids":
                return generated_ids[0]
            return output_text

        finally:
            # restore vanilla visual forward function
            visual.forward = original_visual_forward


if __name__ == "__main__":
    # Comparison example: original qwen VS Our API
    # They should output exactly the same sentence.
    
    MODEL_PATH = "/path/to/your/Qwen3VL-8B-Instruct"
    DEVICE = "cuda:5"
    IMAGE_PATH = "/path/to/a/test/image.jpg"
    PROMPT = "Briefly describe this image."
    
    img = Image.open(IMAGE_PATH).convert("RGB")
    
    # ======= VANILLA QWEN3VL ======= #
    print("RUNNING VANILLA QWEN3VL...")
    vanilla_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map=DEVICE,
        attn_implementation="eager",
        dtype="bfloat16"
    )
    vanilla_model.eval()
    
    vanilla_processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        min_pixels=768*28*28,
        max_pixels=1536*28*28
    )
    
    vanilla_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    
    vanilla_inputs = vanilla_processor.apply_chat_template(
        vanilla_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(DEVICE)
    
    
    with torch.inference_mode():
        vanilla_outputs = vanilla_model.generate(
            **vanilla_inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=vanilla_processor.tokenizer.pad_token_id,
            eos_token_id=vanilla_processor.tokenizer.eos_token_id,
        )
    
    vanilla_generated_ids = vanilla_outputs[0][len(vanilla_inputs["input_ids"][0]):]
    vanilla_prediction = vanilla_processor.decode(vanilla_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    print("VANILLA's OUTPUT:")
    print(vanilla_prediction)
    
    del vanilla_model
    torch.cuda.empty_cache()
    
    # ======= OUR API ======= #
    print("\nRUNNING OUR API...")
    wrapper = Qwen3VLWrapper(model_path=MODEL_PATH, device=DEVICE,dtype="bfloat16")
    
    pixel_values, image_grid_thw = wrapper.prepare_image(img)
    
    hidden_states = wrapper.extract_features(pixel_values, image_grid_thw)
    print(f"  hidden_states shape (after extract): {hidden_states.shape}") # <- NOTE: Apply compression to this.
    
    ### EXAMPLE ###
    # bits = your.compress(hidden_states)
    # hidden_states_hat = your.decompress(bits)
    ###############
    
    ctc_prediction = wrapper.generate_from_features(
        hidden_states,
        prompt=PROMPT,
        max_new_tokens=512,
        do_sample=False,
    )
    
    print("OUR API's OUTPUT:")
    print(ctc_prediction)
    
    
        