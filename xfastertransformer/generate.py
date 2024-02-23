import xfastertransformer
import importlib.util
import argparse
from transformers import AutoTokenizer, TextStreamer
import time
import torch
import os
from typing import Tuple, List

# Ignore Tensor-RT warning from huggingface
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return low_string == "true"


DTYPE_LIST = ["fp16", "bf16", "int8", "int4",
              "bf16_fp16", "bf16_int8", "bf16_int4"]
bc_token = "/home/ge/models/Baichuan2-13B-Base"
bc_model = "/home/ge/models/baichuan2-13b-xft-fp16"
qw_dir = "/home/ge/models/qwen-7b-chat-xft"
parser = argparse.ArgumentParser()
parser.add_argument("--token_path", type=str,
                    default=qw_dir, help="Path to token file")
parser.add_argument("--model_path", type=str,
                    default=qw_dir, help="Path to model file")
parser.add_argument("--dtype", type=str, choices=DTYPE_LIST,
                    default="fp16", help="Data type")
parser.add_argument("--padding", help="Enable padding, Default to True.",
                    type=boolean_string, default=False)
parser.add_argument("--streaming", help="Streaming output, Default to True.",
                    type=boolean_string, default=False)
parser.add_argument(
    "--num_beams", help="Num of beams, default to 1 which is greedy search.", type=int, default=1)
parser.add_argument(
    "--output_len", help="max tokens can generate excluded input.", type=int, default=100)
parser.add_argument("--chat", help="Enable chat mode, Default to False.",
                    type=boolean_string, default=False)
parser.add_argument("--do_sample", help="Enable sampling search, Default to False.",
                    type=boolean_string, default=False)
parser.add_argument(
    "--temperature", help="value used to modulate next token probabilities.", type=float, default=1.0)
parser.add_argument(
    "--top_p", help="retain minimal tokens above topP threshold.", type=float, default=1.0)
parser.add_argument(
    "--top_k", help="num of highest probability tokens to keep for generation", type=int, default=50)


def build_inputs_chatglm(tokenizer, query: str, padding, history: List[Tuple[str, str]] = []):
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(
            i + 1, old_query, response)
    prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
    inputs = tokenizer(prompt, return_tensors="pt", padding=padding).input_ids
    return inputs


def build_inputs_baichuan(tokenizer, query: str, padding, history: List[Tuple[str, str]] = []):
    inputs = tokenizer(query, return_tensors="pt", padding=padding).input_ids
    suffix = torch.tensor([[196]])
    prefix = torch.tensor([[195]])
    inputs = torch.cat((prefix, inputs, suffix), dim=1)
    return inputs


# xft_spec = importlib.util.find_spec("xfastertransformer")

# if xft_spec is None:
#     import sys

#     sys.path.append("../../src")
#     print("[INFO] xfastertransformer is not installed in pip, using source code.")
# else:
#     print("[INFO] xfastertransformer is installed, using pip installed package.")

def generate_interactive(args):
    tokenizer = AutoTokenizer.from_pretrained(args.token_path, use_fast=False, padding_side="left", trust_remote_code=True)
    import automodel
    model = automodel.AutoModel.from_pretrained(args.model_path, dtype=args.dtype)
    streamer = None
    if model.rank == 0 and args.streaming and args.num_beams == 1:
        streamer = TextStreamer(
            tokenizer, skip_special_tokens=True, skip_prompt=False)
    if model.rank == 0:
        # Master
        while True:
            input_prompt = input("\nPlease enter the prompt: ")
            if input_prompt == "":
                input_prompt = DEFAULT_PROMPT
                print("[Use default prompt]:" + input_prompt)
            if args.chat and "chatglm" in args.model_path.lower():
                input_ids = build_inputs_chatglm(
                    tokenizer, input_prompt, args.padding)
            elif "baichuan" in args.model_path.lower():
                input_ids = build_inputs_baichuan(
                    tokenizer, input_prompt, args.padding)
            else:
                input_ids = tokenizer(
                    input_prompt, return_tensors="pt", padding=args.padding).input_ids
            # print("=" * 50)

            perf = {"latency": []}
            start_time = time.perf_counter()
            generated_ids = model.generate(
                input_ids,
                max_length=input_ids.shape[-1] + args.output_len,
                streamer=streamer,
                num_beams=args.num_beams,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                perf=perf,
            )
            end_time = time.perf_counter()

            if streamer is None:
                ret = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True)
                for snt in ret:
                    print(snt)
            print("=" * 20 + "Performance" + "=" * 20)
            execution_time = end_time - start_time
            print(f"Execution time:\t{execution_time:.2f} s")
            input_token_nums = torch.numel(input_ids)
            output_token_nums = torch.numel(generated_ids) - input_token_nums
            latency = execution_time * 1000 / output_token_nums
            througput = output_token_nums / execution_time
            print(f"Latency:\t{latency:.2f} ms/token")
            print(f"Througput:\t{througput:.2f} tokens/s")

            latency = perf["latency"]
            # print(latency)

            result = {
                 "total_token_latency_s": sum(latency),
                # first token completion latency
                "first_token_latency_ms": latency[0]*1000 if len(latency) > 0 else 0,
                # next token completion latency
                "next_token_latency_ms": sum(latency[1:])*1000 / len(latency[1:]) if len(latency) > 1 else 0,
                # average token completion latency
                "avg_token_latency_ms": sum(latency)*1000 / len(latency) if len(latency) > 0 else 0,
            }
            print(result)
    else:
        # Slave
        while True:
            model.generate()

def generate(args, input_prompt):
    # import os
    # os.environ("CCL_LOG_LEVEL")="info"
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()
    # name = MPI.Get_processor_name()
    tokenizer = AutoTokenizer.from_pretrained(args.token_path, use_fast=False, padding_side="left", trust_remote_code=True)
    import automodel
    model = automodel.AutoModel.from_pretrained(args.model_path, dtype=args.dtype)
    streamer = None
    if model.rank == 0 and args.streaming and args.num_beams == 1:
        streamer = TextStreamer(
            tokenizer, skip_special_tokens=True, skip_prompt=False)
    if model.rank == 0:
        print(f"master: rank-{model.rank}", flush=True)
        # Master
        if args.chat and "chatglm" in args.model_path.lower():
            input_ids = build_inputs_chatglm(
                tokenizer, input_prompt, args.padding)
        elif "baichuan" in args.model_path.lower():
            input_ids = build_inputs_baichuan(
                tokenizer, input_prompt, args.padding)
        else:
            input_ids = tokenizer(
                input_prompt, return_tensors="pt", padding=args.padding).input_ids

        perf = {"latency": []}
        start_time = time.perf_counter()
        generated_ids = model.generate(
            input_ids,
            max_length=input_ids.shape[-1] + args.output_len,
            streamer=streamer,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            perf=perf,
        )
        end_time = time.perf_counter()

        if streamer is None:
            ret = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True)
            for snt in ret:
                print(snt)
        # print("=" * 20 + "Performance" + "=" * 20)
        execution_time = end_time - start_time
        print(f"Execution time:\t{execution_time:.2f} s")
        input_token_nums = torch.numel(input_ids)
        output_token_nums = torch.numel(generated_ids) - input_token_nums
        latency = execution_time * 1000 / output_token_nums
        througput = output_token_nums / execution_time
        print(f"Latency:\t{latency:.2f} ms/token")
        print(f"Througput:\t{througput:.2f} tokens/s")

        latency = perf["latency"]

        result = {
            "total_token_latency_s": sum(latency),
            # first token completion latency
            "first_token_latency_ms": latency[0]*1000 if len(latency) > 0 else 0,
            # next token completion latency
            "next_token_latency_ms": sum(latency[1:])*1000 / len(latency[1:]) if len(latency) > 1 else 0,
            # average token completion latency
            "avg_token_latency_ms": sum(latency)*1000 / len(latency) if len(latency) > 0 else 0,
        }
        print(f"Result:\n{result}")
        # comm.send(result, dest=0)
    else:
        # Slave
        print(f"worker: rank-{model.rank}", flush=True)
        model.generate()

DEFAULT_PROMPT = "Once upon a time, there existed a little girl who liked to have adventures."

# from mpi4py import MPI
# from mpi4py.futures import MPIPoolExecutor
if __name__ == "__main__":
    args = parser.parse_args()
    generate(args, DEFAULT_PROMPT)
    # with MPIPoolExecutor(max_workers=2) as executor:
    #     # i=0
    #     # for result in executor.map(generate, args, DEFAULT_PROMPT, MPI):
    #     #     print(f"result-{i}:\nresult")
    #     #     i+=1
    #     future = executor.submit(generate, args, DEFAULT_PROMPT)
    #     print(f"result:\n{future.result()}")

    
    
