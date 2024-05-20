import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path="/home/ge/models/Mixtral-8x7B-Instruct-v0.1.safetensors"
model2="/models/Mixtral-8x7B-Instruct-v0.1.safetensors"
model = AutoModelForCausalLM.from_pretrained(model2, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model2, trust_remote_mode=True)

warm_prompt = "My favourite condiment is"
device="cpu"
model_inputs = tokenizer([warm_prompt], return_tensors="pt").to(device)
model.to(device)
print("warmup")
generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)

print("bench")
bench_prompt="generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)"
model_inputs = tokenizer([bench_prompt], return_tensors="pt").to(device)
print("Input len: ", len(model_inputs[0]))
import time
st=time.perf_counter()
generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
end=time.perf_counter()
print(tokenizer.batch_decode(generated_ids)[0])
gen_len = len(generated_ids[0])
print("Total generation time: ", (end-st)/1000)
print("Gen len: ", gen_len)
print("avg latency(ms/token): ", (end-st)/gen_len)
print("tps(tokens/s): ", gen_len/((end-st)/1000))
