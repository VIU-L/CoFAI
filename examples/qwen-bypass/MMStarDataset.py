import pandas as pd
import base64
import io
from typing import Dict, Any, Optional
from PIL import Image
import os

class MMStar:
    """
    Loads MMStar Benchmark dataset from a TSV file.
    """

    def __init__(self, data_path: str = "MMStar.tsv",prompt_suffix: Optional[str] = ""):
        if not os.path.exists(data_path):
            raise FileNotFoundError("Data TSV not found. Please download from: https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv and place it in the current directory.")
        
        self.data = pd.read_csv(data_path, sep="\t")
        self.data["index"] = self.data["index"].astype(str)
        self.option_cols = ["A", "B", "C", "D"]
        self.prompt_suffix = prompt_suffix
        
    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def _decode_image(image_str: str) -> Image.Image:
        """
        Decodes a base64-encoded image string into a PIL Image object.
        """
        if image_str.startswith("data:image"):
            image_str = image_str.split(",", 1)[1]
        image_bytes = base64.b64decode(image_str)
        return Image.open(io.BytesIO(image_bytes))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dictionary containing the following keys:
            {
                "index": str,               # sample ID
                "image": PIL.Image.Image,   # PIL Image object
                "question": str,            # question text (excluding hints)
                "options": dict,            #  {'A': '...', 'B': '...', ...}
                "answer": str,              # correct option letter (e.g., 'A')
                "category": str,            # primary category
                "l2_category": str,         # secondary category
                "bench": str,               # source benchmark
                "prompt": str               # constructed multimodal prompt text (with question + options)
            }
        """
        item = self.data.iloc[idx]
        options = {c: item[c] for c in self.option_cols if pd.notna(item[c])}
        prompt_parts = []
        
        
        # Delete "Hint:" line (no use) from the question text
        question_lines = item["question"].splitlines()
        filtered_question=""
        for line in question_lines:
            if line.strip().startswith("Hint:"):
                continue
            filtered_question += line + "\n"
        filtered_question = filtered_question.strip()
        prompt_parts.append(f"Question: {filtered_question}")
                
        
        if options:
            opt_str = "\n".join([f"{k}. {v}" for k, v in options.items()])
            prompt_parts.append("Options:\n" + opt_str + "\n")
            prompt_parts.append(self.prompt_suffix)
        prompt = "\n".join(prompt_parts)

        # Decode image to PIL
        image = self._decode_image(str(item["image"]))

        result = {
            "index": str(item["index"]),
            "image": image,                       
            "question": str(item["question"]),
            "options": options,
            "answer": str(item["answer"]),
            "category": str(item["category"]) if pd.notna(item["category"]) else None,
            "l2_category": str(item["l2_category"]) if pd.notna(item["l2_category"]) else None,
            "bench": str(item["bench"]) if "bench" in item and pd.notna(item["bench"]) else None,
            "prompt": prompt,
        }
        return result

