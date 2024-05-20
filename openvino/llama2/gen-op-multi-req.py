import openvino as ov
from transformers import LlamaTokenizer
from optimum.intel.openvino import OVModelForCausalLM
import time
import json
import argparse
import logging as log
import sys
log.basicConfig(format='[ %(levelname)s ] %(message)s',
                level=log.INFO, stream=sys.stdout)

parser = argparse.ArgumentParser(add_help=False)


def generate(model, inputs, args, perf):
    st = time.perf_counter()
    output_ids = model.generate(inputs.input_ids,
                            max_length=args.max_sequence_length+prompt_tokens,
                            pad_token_id = 2,
                            attention_mask = inputs["attention_mask"],
                            perf=perf)
    end = time.perf_counter()
    # queue.put(f'dur:{end-st}')
    perf["dur_s"] = (end-st)/1000
    perf["output_ids"] = output_ids

def get_result(perf):
    output_ids = perf["output_ids"]
    completion_ids = [output_id.tolist()[prompt_tokens:] for output_id in output_ids]
    log.info(" ==================== text decoding ==================== ")
    completions = tokenizer.batch_decode(
                                sequences=completion_ids,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
    latency = perf["latency"]
    log.info(f'completions: {completions}')
    log.info(f'prompt_tokens: {prompt_tokens}')
    log.info(f'batch_size: {inputs.input_ids.shape[0]}')
    log.info(f'total_dur_s: {perf["dur_s"]:.3f}')  # total time, include tokeninzer.encode+decode, tokens generation
    log.info(f'completion_tokens: {len(completion_ids)}')
    log.info(f'total_token_latency_s: {sum(latency)}')
    log.info(f'first_token_latency_ms: {latency[0]*1000 if len(latency) > 0 else 0}')        # next token completion latency
    log.info(f'next_token_latency_ms: {sum(latency[1:])*1000 / len(latency[1:]) if len(latency) > 1 else 0}')
    log.info(f'avg_token_latency_ms: {sum(latency)*1000 / len(latency) if len(latency) > 0 else 0}')

parser.add_argument('-h',
                    '--help',
                    action='help',
                    help='Show this help message and exit.')
parser.add_argument('-m',
                    '--model_id',
                    required=False,
                    type=str,
                    default='/home/ge/models/llama2-7b-ov',
                    help='Required. hugging face model id or local model path')
parser.add_argument('-p',
                    '--prompt',
                    default='What is AI?',
                    type=str,
                    help='prompt sentence')
parser.add_argument('-l',
                    '--max_sequence_length',
                    default=128,
                    required=False,
                    type=int,
                    help='maximun lengh of output')
parser.add_argument('-d',
                    '--device',
                    default='CPU',
                    required=False,
                    type=str,
                    help='device for inference')
parser.add_argument('-a',
                    '--accuracy-mode',
                    default=False,
                    required=False,
                    type=bool,
                    help='')
parser.add_argument('-t',
                    '--num-threads',
                    default=False,
                    required=False,
                    type=int,
                    help='inference threads number')
parser.add_argument('-n',
                    '--num-streams',
                    default=2,
                    required=False,
                    type=int,
                    help='streams number')
parser.add_argument('-j',
                    '--prompt-path',
                    default='../prompt.json',
                    required=False,
                    type=str,
                    help='Path to prompt file')
parser.add_argument('-b',
                    '--batch-size',
                    default=1,
                    required=False,
                    type=int,
                    help='batch size')
args = parser.parse_args()

log.info(" ==================== prepare prompt ==================== ")
with open(args.prompt_path, "r") as prompt_file:
    prompt_pool = json.load(prompt_file)
    model_prompt=prompt_pool["llama"]
if args.batch_size == 1:
    input_prompts = args.prompt
else:
    input_prompts = [model_prompt["22"] for i in range(args.batch_size)]
log.info(f'prompts: {input_prompts}')

log.info(" ==================== load tokenizer ==================== ")
tokenizer = LlamaTokenizer.from_pretrained(
    args.model_id, trust_remote_code=True)

core = ov.Core()

# load model
ov_config = {
    # 'PERFORMANCE_HINT': 'LATENCY',
    # 'INFERENCE_NUM_THREADS': int(args.num_threads),
    # ov.properties.num_streams(): args.num_streams,
    "CACHE_DIR": "./",
}
try:
    log.info(" ==================== use local model ==================== ")
    model = OVModelForCausalLM.from_pretrained(
        args.model_id,
        compile=False,
        device=args.device,
        ov_config=ov_config,
    )
except:
    log.info(" ==================== use remote model ==================== ")
    model = OVModelForCausalLM.from_pretrained(
        args.model_id, compile=False, device=args.device, export=True)

model.compile()

# get property
compiled_model = model.request.get_compiled_model()
log.info(f'PERFORMANCE_HINT: {compiled_model.get_property("PERFORMANCE_HINT")}')
log.info(f'inference_num_threads: {compiled_model.get_property("inference_num_threads".upper())}')
log.info(f'enable_hyper_threading: {compiled_model.get_property("enable_hyper_threading".upper())}')
log.info(f'num_streams: {compiled_model.get_property(ov.properties.num_streams())}')
log.info(f'OPTIMAL_NUMBER_OF_INFER_REQUESTS: {compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")}')
log.info(f'inference_precision: {compiled_model.get_property(ov.properties.hint.inference_precision())}')

log.info(" ==================== Warm up ====================")
warm_prompt = "Once upon a time, there existed a little girl who liked to have adventures."
warm_inputs = tokenizer(warm_prompt, return_tensors="pt")
warm_prompt_tokens = warm_inputs.input_ids.shape[1]
# log.info(type(warm_inputs.input_ids))
# log.info(warm_inputs.input_ids)
# log.info(warm_inputs.input_ids.shape[0])
# log.info(warm_inputs.input_ids.shape[1])
perf = {"latency": []}
_ = model.generate(warm_inputs.input_ids,
                    max_length=args.max_sequence_length+warm_prompt_tokens,
                    pad_token_id = 2,
                    attention_mask = warm_inputs["attention_mask"],
                    perf=perf)
# if args.batch_size > 1:
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side='left'
inputs = tokenizer(input_prompts,
                    return_tensors="pt",
                    padding=True)
# if args.num_streams == 1:
#     inputs = tokenizer(args.prompt, return_tensors="pt")
# elif args.num_streams == 2:
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side='left'
#     inputs = tokenizer([args.prompt, "once upon a time, there existed a little girl who liked to have"],
#                         return_tensors="pt",
#                         padding=True)
# inputs = tokenizer(args.prompt, return_tensors="pt")
# log.info(f'input_ids: {type(inputs.input_ids)}')
# log.info(inputs.input_ids)
# log.info(inputs.input_ids.shape[0])
# log.info(inputs.input_ids.shape[1])

prompt_tokens = inputs.input_ids.shape[1]
log.info(" ==================== start generating in paralle ==================== ")
import threading
perf1 = {"latency": []}
perf2 = {"latency": []}
# from queue import Queue
# result_queue = Queue()
thread1 = threading.Thread(target=generate, args=(model,inputs,args, perf1))
thread2 = threading.Thread(target=generate, args=(model,inputs,args, perf2))
# 启动线程
thread1.start()
thread2.start()

# 等待两个线程执行完毕
thread1.join()
thread2.join()

log.info("==================== Both threads have finished ======================")

log.info(("==================== result of thread1 ======================"))

get_result(perf1)

log.info(("==================== result of thread2 ======================"))
get_result(perf2)