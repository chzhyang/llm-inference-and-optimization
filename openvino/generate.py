from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
tokenizer = AutoTokenizer.from_pretrained("/home/ge/models/Qwen-7B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/home/ge/models/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()

response, history, _ = model.chat(tokenizer, "你好", history=None)
print(response)
response, history, _ = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
print(response)
response, history, perf = model.chat(tokenizer, "给这个故事起一个标题", history=history)
print(response, perf)