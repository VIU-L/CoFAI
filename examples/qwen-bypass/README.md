# Qwen3VL-8B-Instruct Eval Code

## Description
This folder provides 2 functionalities:  

1. Fast evaluation of Qwen3VL-8B-Instruct on MMstar; result ("Bypass" accuracy) is 67.40%, and can be viewed in folder `MMStar-CTC`.
The model is asked to produce single-letter predictions, thus runs very fast (evaluation only takes 11 minutes).  
Peak GPU memory usage is ~19GB.  

2. API to (1) extract `hidden_states` at Slot 9 of Visual Encoder, (2) put it back and continue generation.
This  `hidden_states` (1152,H,W) is the object to be compressed/decompressed by your codec.  
Without compression, this API replicates the exact behaviour of the original Qwen3VL.

## Requirements
The script and the API are tested on `torch==2.6.0` and `transformers==5.12.1`.  
They should theoretically work on earlier versions of `transformers`, as long as Qwen3VL is supported by that version.

Before running evaluation, you should download `MMStar.tsv` from [Openxlab](https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv), and put it in this folder.