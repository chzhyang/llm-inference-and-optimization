import os
import sys

import torch
import transformers
import peft
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
device = "cpu"

def merge(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "",
    merge_lora_model: bool = False,
    merged_model_path: str = "",
):
    base_model = base_model or os.environ.get("BASE_MODEL", "/home/ge/models/Llama-2-7b-hf")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='llama/llama-7b'"
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model, 
        # device_map={"": device}, 
        # low_cpu_mem_usage=True, 
        # torch_dtype=torch.bfloat16,
        torch_dtype=torch.float16,
    )
    print("## evaluate base model")
    evaluate(model, tokenizer)
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        # device_map={"": device},
        # torch_dtype=torch.bfloat16,
        torch_dtype=torch.float16,
    )

    if merge_lora_model:
        print("## evaluate lora befor merge")
        evaluate(model, tokenizer)
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        if hasattr(peft.LoraModel,'merge_and_unload'):
            print("## begin to save merged lora model")
            merged_model = model.merge_and_unload()
            # default saved to safetensors
            LlamaForCausalLM.save_pretrained(merged_model, merged_model_path, safe_serialization=False) 
            # merged_model.save_pretrained(merged_model_path, safe_serialization=False)
            tokenizer.save_pretrained(merged_model_path)
            print("## saved merged lora model")
            print("## evaluate after merge")
            from transformers import LlamaConfig
            print("/t## 1")
            config_path = LlamaConfig.from_pretrained(base_model)
            print("/t## 2")
            model = LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=merged_model_path, 
                config=config_path, 
                torch_dtype=torch.float16,
                use_safetensors=False,
            )
            print("/t## 3")
            tokenizer = LlamaTokenizer.from_pretrained(merged_model_path)
            print("/t## 4")
            evaluate(model=model,tokenizer=tokenizer)
        else:
            print("no support for merging model")

def evaluate(model, tokenizer):
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    for prompt in [
        "What is the relationship between very low Mg2+ levels, PTH levels, and Ca2+ levels?",
        # "What does low REM sleep latency and experiencing hallucinations/sleep paralysis suggest?",
    ]:
        print("prompt:", prompt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output_ids = model.generate(
            input_ids,
            do_sample=True,
            max_new_tokens=128,
        )
        prompt_tokens = input_ids.shape[1]
        completion_ids = output_ids[0].tolist()[prompt_tokens:]
        completion = tokenizer.decode(
            completion_ids, skip_special_tokens=True)
        print("output: ", completion)

if __name__ == "__main__":
    # merge(
    #     base_model="/home/ge/models/Llama-2-7b-hf",
    #     lora_weights="/home/ge/models/medical_llama2_7b",
    #     merge_lora_model=True,
    #     merged_model_path="/home/ge/models/merged_medical_llama2_7b",
    # )

    # main(
    #     base_model="/home/ge/models/Llama-2-7b-hf",
    #     lora_weights="/home/ge/models/medical_llama2_7b",
    #     merge_lora_model=False,
    #     merged_model_path="/home/ge/models/merged_medical_llama2_7b",
    # )
    
    # https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/scripts
    # https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/merge_llama_with_chinese_lora.py
    # https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/inference/gradio_demo.py#L90

    print("## evaluate merged lora")
    from transformers import LlamaConfig
    # print("/t## 1")
    # config_path = LlamaConfig.from_pretrained("/home/ge/models/merged_medical_llama2_7b") # replaced config.json by Llama2's config
    print("/t## 2")
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path="/home/ge/models/merged_medical_llama2_7b", 
        # config=config_path, 
        torch_dtype=torch.float16,
        use_safetensors=False,
    )
    print("/t## 3")
    tokenizer = LlamaTokenizer.from_pretrained("/home/ge/models/merged_medical_llama2_7b")
    print("/t## 4")
    evaluate(model=model,tokenizer=tokenizer)