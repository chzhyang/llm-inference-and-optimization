import torch
import time
import intel_extension_for_pytorch as ipex
import transformers
from transformers import AutoTokenizer
model_id = "/home/ge/models/Llama-2-7b-hf"
print("load")
model= transformers.AutoModelForCausalLM.from_pretrained(model_id).eval()
print("quantize")
qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
  weight_dtype=torch.qint8, # or torch.quint4x2
  lowp_mode=ipex.quantization.WoqLowpMode.NONE, # or FP16, BF16, INT8
)

import warnings
# 关闭所有警告
warnings.filterwarnings("ignore")
checkpoint = None # optionally load int4 or int8 checkpoint
model = ipex.optimize_transformers(model, quantization_config=qconfig, low_precision_checkpoint=checkpoint)
for name, param in model.named_parameters():
    print(f"{name}: {param.dtype}")
# 恢复警告
warnings.resetwarnings()
tokenizer = AutoTokenizer.from_pretrained(
                           model_id, trust_remote_code=True)
prompt="what is ai?"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
with torch.inference_mode():
    print("generate")
    st = time.time()
    output_ids=model.generate(input_ids, max_new_tokens=100)
    prompt_tokens = input_ids.shape[1]
    completions_ids = output_ids[0].tolist()[prompt_tokens:]
    out = tokenizer.decode(completions_ids, skip_special_tokens=True)
    end = time.time()
    generate_tokens = len(completions_ids)
    avg_token_latency = (end-st)*1000/generate_tokens
    print(out)
    print("tokens:", generate_tokens, "avg_latency:",avg_token_latency)
del model
import gc
gc.collect()


