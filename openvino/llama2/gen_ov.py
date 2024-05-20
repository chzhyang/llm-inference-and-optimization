from typing import Optional
from transformers import LlamaTokenizer
import openvino as ov
from openvino.runtime import Core, Tensor
from pathlib import Path
import numpy as np
import argparse
import time
import logging as log
import sys
log.basicConfig(format='[ %(levelname)s ] %(message)s',
                level=log.INFO, stream=sys.stdout)

def param_to_string(parameters) -> str:
        """Convert a list / tuple of parameters returned from IE to a string."""
        if isinstance(parameters, (list, tuple)):
            return ', '.join([str(x) for x in parameters])
        else:
            return str(parameters)

def read_config():
    log.info('Available devices:')
    for device in core.available_devices:
        log.info(f'{device}:')
        log.info('\tSUPPORTED_PROPERTIES:')
        for property_key in core.get_property(device, 'SUPPORTED_PROPERTIES'):
            if property_key not in ('SUPPORTED_METRICS', 'SUPPORTED_CONFIG_KEYS', 'SUPPORTED_PROPERTIES'):
                try:
                    property_val = core.get_property(device, property_key)
                except TypeError:
                    property_val = 'UNSUPPORTED TYPE'
                log.info(f'\t\t{property_key}: {param_to_string(property_val)}')
        log.info('')

def get_model_properties(model: ov.runtime.CompiledModel):
    log.info('Available devices:')
    for device in core.available_devices:
        log.info(f'{device}:')
        log.info('\tSUPPORTED_PROPERTIES:')
        for property_key in core.get_property(device, 'SUPPORTED_PROPERTIES'):
            if property_key not in ('SUPPORTED_METRICS', 'SUPPORTED_CONFIG_KEYS', 'SUPPORTED_PROPERTIES'):
                try:
                    property_val = model.get_property(device, property_key)
                except TypeError:
                    property_val = 'UNSUPPORTED TYPE'
                log.info(f'\t\t{property_key}: {param_to_string(property_val)}')
        log.info('')

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    summation = e_x.sum(axis=-1, keepdims=True)
    return e_x / summation


def process_logits(cur_length, scores, eos_token_id, min_length=0):
    if cur_length < min_length:
        scores[:, eos_token_id] = -float("inf")
    return scores


def get_top_k_logits(scores, top_k):
    filter_value = -float("inf")
    top_k = min(max(top_k, 1), scores.shape[-1])
    top_k_scores = -np.sort(-scores)[:, :top_k]
    indices_to_remove = scores < np.min(top_k_scores)
    filtred_scores = np.ma.array(scores,
                                 mask=indices_to_remove,
                                 fill_value=filter_value).filled()
    return filtred_scores


def generate_sequence(sampling, input_ids, attention_mask, eos_token_id,
                      max_sequence_length, perf: Optional[dict]=None):
    if perf is None:
        perf = {"latency":[]}
    latency = perf["latency"]
    st = time.perf_counter()

    past_key_values = None
    prompt_len = len(input_ids[0])
    count = 0

    while True:
        try:
            inputs = {}
            if past_key_values is not None:
                inputs = dict(zip(key_value_input_names, past_key_values))
                inputs["input_ids"] = input_ids[:, -1:]
                cur_input_len = 1
            else:
                inputs["input_ids"] = input_ids
                shape_input_ids = input_ids.shape
                num_attention_heads = 1
                for input_name in key_value_input_names:
                    model_inputs = model.input(input_name)
                    shape = model_inputs.get_partial_shape()
                    shape[0] = shape_input_ids[0] * num_attention_heads
                    if shape[2].is_dynamic:
                        shape[2] = 0
                    if shape[1].is_dynamic:
                        shape[1] = 0
                    inputs[input_name] = Tensor(model_inputs.get_element_type(),
                                                shape.get_shape())
            cur_input_len = len(inputs["input_ids"][0])
            if "attention_mask" in input_names and attention_mask is not None:
                inputs["attention_mask"] = attention_mask
            request.start_async(inputs, share_inputs=True)
            request.wait()
            count += 1
            logits = request.get_tensor("logits").data
            past_key_values = tuple(
                request.get_tensor(key).data for key in key_value_output_names)
            next_token_logits = logits[:, cur_input_len - 1, :]
            # pre-process distribution
            next_token_scores = process_logits(len(input_ids[0]),
                                               next_token_logits, eos_token_id)
            top_k = 20
            next_token_scores = get_top_k_logits(next_token_scores, top_k)
            # get next token id
            if sampling:
                probs = softmax(next_token_scores)
                next_tokens = np.random.choice(probs.shape[-1],
                                               1,
                                               p=probs[0],
                                               replace=True)
            else:
                next_tokens = np.argmax(next_token_scores, axis=-1) # greedy search
            # break the loop if max length or end of text token is reached
            if (len(input_ids[0]) - prompt_len
                    ) == max_sequence_length or next_tokens == eos_token_id:
                # end = time.perf_counter()
                # latency.append(end - st)
                break
            else:
                input_ids = np.concatenate((input_ids, [next_tokens]), axis=-1)
                attention_mask = np.concatenate(
                    (attention_mask, [[1] * len(next_tokens)]), axis=-1)

        finally:
            end = time.perf_counter()
            latency.append(end - st)
            st = end
            # tmp = input_ids[0].tolist()
            # print(len(latency), len(tmp), tmp)
    return input_ids, count

def warmup():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h',
                        '--help',
                        action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-m',
                        '--model_path',
                        required=False,
                        default="/home/ge/models/llama2-7b-ov",
                        type=str,
                        help='Required. path of IR model and tokenizer')
    parser.add_argument('-p',
                        '--prompt',
                        type=str,
                        default="What is AI?",
                        help='Required. prompt sentence')
    
    parser.add_argument('-n',
                        '--num-streams',
                        default=1,
                        required=False,
                        type=int,
                        help='streams number')
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
    parser.add_argument('-s',
                        '--sampling',
                        default=False,
                        required=False,
                        type=bool,
                        help='sampling or not, use greedy search for bench, random sampling for chat')
    args = parser.parse_args()

    num_pkv = 2
    core = Core()
    ir_model_path = Path(args.model_path)
    ir_model = ir_model_path / "openvino_model.xml"

    print(" --- load tokenizer --- ")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    if args.num_streams == 1:
        inputs = tokenizer(args.prompt, return_tensors="np")
    elif args.num_streams == 2:
        inputs = tokenizer([args.prompt, args.prompt], return_tensors="np", padding=True)

    print("inputs:", type(inputs), inputs)
    print("inputs[input_ids]:", type(inputs["input_ids"]), inputs["input_ids"])

    print(" --- read model --- ")
    # read the model and corresponding weights from file
    model = core.read_model(ir_model)
    input_names = {
        key.get_any_name(): idx
        for idx, key in enumerate(model.inputs)
    }
    output_names = {
        key.get_any_name(): idx
        for idx, key in enumerate(model.outputs)
    }
    key_value_input_names = [key for key in input_names if "key_values" in key]
    key_value_output_names = [key for key in output_names if "present" in key]

    import os
    num_cores = os.cpu_count()
    log.info(f'num_cores: {num_cores}')
    

    # core.set_property("CPU",{"INFERENCE_NUM_THREADS": 230})
    # core.set_property("CPU",{"CPU_BIND_THREAD": "NUMA"})
    core.set_property("CPU",{ov.properties.hint.enable_hyper_threading(): False})
    read_config()

    print(" --- model compiling --- ")
    # compile the model for CPU devices
    compile_model = core.compile_model(
        model=model, 
        device_name=args.device,
        config={
            # ov.properties.inference_num_threads(): 120,
            ov.properties.cache_dir(): "./model_cache",
            ov.properties.num_streams(): args.num_streams,
            ov.properties.hint.enable_hyper_threading(): False,
            }
    )
    log.info(f'threads: {compile_model.get_property(ov.properties.inference_num_threads())}')
    log.info(f'enable_hyper_threading: {compile_model.get_property(ov.properties.hint.enable_hyper_threading())}')
    log.info(f'num_streams: {compile_model.get_property(ov.properties.num_streams())}')
    log.info(f'inference_precision: {compile_model.get_property(ov.properties.hint.inference_precision())}')

    request = compile_model.create_infer_request()

    perf = {"latency": []}
    print(" --- warm up ---")
    warm_prompt = "Once upon a time, there existed a little girl who liked to have adventures."
    warm_inputs = tokenizer(warm_prompt, return_tensors="np")
    _,_ = generate_sequence(
        args.sampling,
        warm_inputs["input_ids"],
        warm_inputs["attention_mask"],
        eos_token_id=tokenizer.eos_token_id,
        max_sequence_length=args.max_sequence_length,
    )
    print(" --- start generating --- ")
    st = time.perf_counter()
    output_ids, num_tokens = generate_sequence(
        args.sampling,
        inputs["input_ids"],
        inputs["attention_mask"],
        eos_token_id=tokenizer.eos_token_id,
        max_sequence_length=args.max_sequence_length,
        perf=perf,
    )
    end = time.perf_counter()

    print("output_ids:", type(output_ids), output_ids)

    latency = perf["latency"]
    print(f'latency len: {len(latency)}')

    output_text = " "
    # Convert IDs to words and make the sentence from it

    print(" --- text decoding --- ")
    # output_text = tokenizer.batch_decode(output_ids,
    #                                      skip_special_tokens=True,
    #                                      clean_up_tokenization_spaces=False)[0]
    print("output_ids[0] len: ", len(output_ids[0]))
    prompt_tokens = inputs.input_ids.shape[1]
    completion_ids = output_ids[0].tolist()[prompt_tokens:]
    completion = tokenizer.decode(completion_ids,
                                  skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)
    print(
        f"Generated {num_tokens} tokens in {end - st:.3f} s on {args.device}"
    )
    # print(f"Response: {output_text}")

    log.info(f'completion: {completion}')
    log.info(f'prompt_tokens: {prompt_tokens}')
    log.info(f'total_dur_s: {end-st}')  # total time, include tokeninzer.encode+decode, tokens generation
    log.info(f'completion_tokens: {len(completion_ids)}')
    log.info(f'total_token_latency_s: {sum(latency)}')
    log.info(f'first_token_latency_ms: {latency[0]*1000 if len(latency) > 0 else 0}')        # next token completion latency
    log.info(f'next_token_latency_ms: {sum(latency[1:])*1000 / len(latency[1:]) if len(latency) > 1 else 0}')
    log.info(f'avg_token_latency_ms: {sum(latency)*1000 / len(latency) if len(latency) > 0 else 0}')
