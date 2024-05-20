import time
import torch
import logging as log
import sys
from typing import List, Tuple

WARMUP_PROMPT = "Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun"

log.basicConfig(format='[ %(levelname)s ] %(message)s',
                    level=log.INFO, stream=sys.stdout)


def llama2_chat_format_prompt(input_str, history):
    LLAMA2_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant, who always answers as helpfully as possible, while being safe."
    prompt = [f'<s>[INST] <<SYS>>\n{LLAMA2_SYSTEM_PROMPT}\n<</SYS>>\n\n']
    do_strip = False
    for history_input, history_response in history:
        history_input = history_input.strip() if do_strip else history_input
        do_strip = True
        prompt.append(
            f'{history_input} [/INST] {history_response.strip()} </s><s>[INST] ')
    input_str = input_str.strip() if do_strip else input_str
    prompt.append(f'{input_str} [/INST]')
    return ''.join(prompt)

class BaseAdapter():
    def __init__(self, model_name, model_dtype, model_path, model_family, n_threads, device, warmup):
        self.model_name = model_name
        self.model_dtype = model_dtype
        self.model_path = model_path
        self.model_family = model_family
        self.n_threads = n_threads
        self.device = device
        self.warmup = warmup

        if self.warmup is None:
            self.warmup = True


    def get_variables(self):        
        log.info(f'\tdevice: {self.device}')
        log.info(f'\tmodel_name: {self.model_name}')
        log.info(f'\tmodel_dtype: {self.model_dtype}')
        log.info(f'\tmodel_path: {self.model_path}')
        log.info(f'\tmodel_family: {self.model_family}')
        log.info(f'\tn_threads: {self.n_threads}')
        log.info(f'\twarmup: {self.warmup}')

class TransformersAdapter(BaseAdapter):
    """
    Huggingface Transformers Adapter

    Support model: models for huggdingface transformers
    Support model datatype: fp32, bf16, int8, int4
    """

    def __init__(self, model_name, model_dtype, model_path, model_family, n_threads, device, warmup, **kwargs):
        super().__init__(
            model_name = model_name,
            model_dtype = model_dtype,
            model_path = model_path,
            model_family = model_family,
            n_threads = n_threads,
            device = device,
            warmup = warmup,
        )
        if model_family == 'llama':
            from transformers import LlamaTokenizer, AutoModelForCausalLM
            log.info(" Loading tokenizer")
            self.tokenizer = LlamaTokenizer.from_pretrained(
                model_path, trust_remote_code=True)
            log.info(" Loading model")
            if model_dtype == "fp32":
                # self.model = AutoModelForCausalLM.from_pretrained(
                #     model_path,
                #     device_map="cpu")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path)
            elif model_dtype == "bf16":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16)
        else:
            from transformers import AutoTokenizer
            log.info("Loading tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True)
            from transformers import AutoModel, AutoModelForCausalLM
            log.info("Loading model")
            if model_dtype == "fp16":
                self.model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True).float()
                # self.model = AutoModel.from_pretrained(
                #     self.model_path,
                #     device_map="cpu",
                #     trust_remote_code=True,
                # ).float()
            elif model_dtype == "bf16":
                self.model = AutoModel.from_pretrained(
                    model_path,
                    device_map="cpu",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True)
            elif model_dtype == "int4":
                self.model = AutoModel.from_pretrained(model_path,
                                                       trust_remote_code=True).float()
        if self.model:
            self.model = self.model.eval()
        if self.warmup:
                self._warmup(WARMUP_PROMPT)

    def _warmup(self, prompt):
        with torch.inference_mode():
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            _ = self.model.generate(input_ids, max_new_tokens=50)

    def create_completion(
            self,
            prompt,
            max_new_tokens,
            top_p, temperature,
            num_beams,
            do_sample,
            stream=False
    ):
        if not stream:
            with torch.inference_mode():
                perf = {"latency": []}
                log.info(f'Start generating')
                st = time.time()
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    perf=perf
                )
                prompt_tokens = input_ids.shape[1]
                completion_ids = output_ids[0].tolist()[prompt_tokens:]
                completion = self.tokenizer.decode(
                    completion_ids, skip_special_tokens=True)
                end = time.time()

                latency = perf["latency"]
                resp = {
                    "completion": completion,
                    "prompt_tokens": prompt_tokens,
                    "total_dur_s": end-st,  # total time, include tokeninzer.encode+decode, tokens generation
                    "completion_tokens": len(completion_ids),
                    # total tokens completion latency, except tokenizer.decode time
                    "total_token_latency_s": sum(latency),
                    # first token completion latency
                    "first_token_latency_ms": latency[0]*1000 if len(latency) > 0 else 0,
                    # next token completion latency
                    "next_token_latency_ms": sum(latency[1:])*1000 / len(latency[1:]) if len(latency) > 1 else 0,
                    # average token completion latency
                    "avg_token_latency_ms": sum(latency)*1000 / len(latency) if len(latency) > 0 else 0,
                }
                return resp
        else:
            # TODO stream
            pass


class BigdlLLMTransformersAdapter(BaseAdapter):
    """
    BigDL-LLM Transformers Adapter

    Support model: most models for huggdingface transformers
    Support model datatype: fp32, model will be converted automatically
    """

    def __init__(self, model_name, model_dtype, model_path, model_family, n_threads, device, warmup, **kwargs):
        super().__init__(
            model_name = model_name,
            model_dtype = model_dtype,
            model_path = model_path,
            model_family = model_family,
            n_threads = n_threads,
            device = device,
            warmup = warmup,
        )
        # model will be converted to int4/int8
        log.info("Loading model")
        from bigdl.llm.transformers import AutoModelForCausalLM
        if model_dtype == "int4":
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_path, load_in_low_bit="sym_int4", trust_remote_code=True)
        elif model_dtype == "int8":
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_path, load_in_low_bit="sym_int8", trust_remote_code=True)

        # load HuggingFace transformers tokenizer
        log.info("Loading tokenizer")
        if model_family == 'llama':
            from transformers import LlamaTokenizer
            self.tokenizer = LlamaTokenizer.from_pretrained(
                model_path, trust_remote_code=True)
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True)
        if self.warmup:
            self._warmup(WARMUP_PROMPT)

    def _warmup(self, prompt):
        log.info(f'Warm up')
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        _ = self.model.generate(input_ids, max_new_tokens=50)

    def create_completion(self, prompt, max_new_tokens, top_p, temperature, do_sample, num_beams, stream=False):
        if not stream:
            perf = {"latency": [], }
            log.info(f'Start generating')
            st = time.time()
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens if max_new_tokens else 2048,
                top_p=top_p if top_p else 0.7,
                temperature=temperature if temperature else 0.95,
                perf=perf)

            prompt_tokens = input_ids.shape[1]
            completion_ids = output_ids[0].tolist()[prompt_tokens:]
            completion = self.tokenizer.decode(
                completion_ids, skip_special_tokens=True)
            end = time.time()

            latency = perf["latency"]
            resp = {
                "completion": completion,
                "prompt_tokens": prompt_tokens,
                "total_dur_s": end-st,  # total time, include tokeninzer.encode+decode, tokens generation
                "completion_tokens": len(completion_ids),
                # total tokens completion latency, except tokenizer.decode time
                "total_token_latency_s": sum(latency),
                # first token completion latency
                "first_token_latency_ms": latency[0]*1000 if len(latency) > 0 else 0,
                # next token completion latency
                "next_token_latency_ms": sum(latency[1:])*1000 / len(latency[1:]) if len(latency) > 1 else 0,
                # average token completion latency
                "avg_token_latency_ms": sum(latency)*1000 / len(latency) if len(latency) > 0 else 0,
            }
            return resp
        else:
            # TODO stream
            pass
        return


class BigdlLLMCPPAdapter(BaseAdapter):
    """
    BigDL-LLM CPP Adapter

    Support model: llama, bloom, gptneox, starcoder
    Support model datatype: int4
    """

    def __init__(self, model_name, model_dtype, model_path, model_family, n_threads, device, warmup, **kwargs):
        super().__init__(
            model_name = model_name,
            model_dtype = model_dtype,
            model_path = model_path,
            model_family = model_family,
            n_threads = n_threads,
            device = device,
            warmup = warmup,
        )
        if model_family == "llama":
            from bigdl.llm.models import Llama
            modelclass = Llama
        # Waiting for bigdl-llm support chatglm
        log.info("Loading model and tokenizer")
        import os
        num_cpus = len(os.sched_getaffinity(0))
        if n_threads is None or n_threads > num_cpus:
            self.n_threads = num_cpus
            
        self.model = modelclass(
            model_path, n_threads=self.n_threads, verbose=True, n_ctx=2048)
        if self.warmup:
            self._warmup(WARMUP_PROMPT)
                

    def _warmup(self, prompt):
        log.info("Warm up")
        _ = self.model(prompt=prompt, max_tokens=50)

    def create_completion(self, prompt, max_new_tokens, top_p, temperature, do_sample, num_beams, stream=False):
        # TODO: latency for stream in _create_completion()
        log.info(f'Start generating')
        st = time.time()
        completion = self.model(prompt=prompt,
                                max_tokens=max_new_tokens,
                                top_p=top_p,
                                temperature=temperature,
                                stream=stream)
        end = time.time()
        resp = {
            "completion": completion["choices"][0]["text"],
            "prompt_tokens": completion["usage"]["prompt_tokens"],
            "completion_tokens": completion["usage"]["completion_tokens"],
            "total_dur_s": end-st,  # total time, include tokeninzer.encode+decode, tokens generation
            # total token completion latency, except tokenizer.decode time
            "total_token_latency_s": completion["usage"]["total_token_latency_s"],
            # first token completion latency
            "first_token_latency_ms": completion["usage"]["first_token_latency_ms"],
            # next token completion latency
            "next_token_latency_ms": completion["usage"]["next_token_latency_ms"],
            # average token completion latency
            "avg_token_latency_ms": completion["usage"]["avg_token_latency_ms"]
        }
        return resp
    # TODO
    # def ctreat_chat_completion(self, messages, max_new_tokens, top_p, temperature, stream=False):
    #     completion = self.model.create_chat_completion(messages=messages, max_tokens=max_new_tokens, top_p=top_p, temperature=temperature, stream=stream)
    #     return completion

import openvino as ov
from openvino.runtime import Core, Tensor
from typing import Optional
import numpy as np

class OpenvinoAdapter(BaseAdapter):
    def __init__(self, model_name, model_dtype, model_path, model_family, n_threads, device, warmup, **kwargs) -> None:
        super().__init__(
            model_name = model_name,
            model_dtype = model_dtype,
            model_path = model_path,
            model_family = model_family,
            n_threads = n_threads,
            device = device,
            warmup = warmup,
        )
        log.info("Loading tokenizer")
        if model_family == 'llama':
            from transformers import LlamaTokenizer
            self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        elif model_family == 'chatglm':
            from transformers import AutoTokenizer
            self.tokenizer =  AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True)

        if model_family in ['llama', 'chatglm'] and model_dtype in ['bf16', 'int8']:
            log.info("Loading model")
            from pathlib import Path
            ir_model = Path(model_path) / "openvino_model.xml"
            core = Core()
            self.model = core.read_model(ir_model)
            self.input_names = {
                key.get_any_name(): idx
                for idx, key in enumerate(self.model.inputs)
            }
            self.output_names = {
                key.get_any_name(): idx
                for idx, key in enumerate(self.model.outputs)
            }
            self.key_value_input_names = [
                key for key in self.input_names if "key_values" in key
            ]
            self.key_value_output_names = [
                key for key in self.output_names if "present" in key
            ]
            ov_config = {
                ov.properties.cache_dir(): "./",
                'PERFORMANCE_HINT': 'LATENCY',
            }
            
            # n_threads must be less than or equal to the number of cpu affinity
            import os
            num_cpus = len(os.sched_getaffinity(0))
            if self.n_threads is not None and self.n_threads > num_cpus:
                self.n_threads = num_cpus
                ov_config[ov.properties.inference_num_threads()] = self.n_threads
                
            compiled_model = core.compile_model(
                model=self.model, 
                device_name=device,
                config=ov_config,
            )

            self.request = compiled_model.create_infer_request()

            if self.warmup:
                self._warmup(WARMUP_PROMPT)
    
    
    def generate_sequence(
            self, 
            sampling,
            input_ids,
            attention_mask,
            eos_token_id,
            max_sequence_length,
            perf: Optional[dict]=None
    ):
        pass
    
    def _generate_sequence_llama(
        self, 
        sampling,
        input_ids,
        attention_mask,
        eos_token_id,
        max_generated_tokens,
        perf: Optional[dict]=None
    ):
        from utils import get_top_k_logits, softmax, process_logits
        if perf is None:
            perf = {"latency":[]}
        latency = perf["latency"]
        st = time.perf_counter()

        past_key_values = None
        prompt_len = len(input_ids[0])

        while True:
            try:
                inputs = {}
                if past_key_values is not None:
                    inputs = dict(zip(self.key_value_input_names, past_key_values))
                    inputs["input_ids"] = input_ids[:, -1:]
                    cur_input_len = 1
                else:
                    inputs["input_ids"] = input_ids
                    shape_input_ids = input_ids.shape
                    num_attention_heads = 1
                    for input_name in self.key_value_input_names:
                        model_inputs = self.model.input(input_name)
                        shape = model_inputs.get_partial_shape()
                        
                        if self.model_family == "llama":
                            shape[0] = shape_input_ids[0] * num_attention_heads
                            if shape[2].is_dynamic:
                                shape[2] = 0
                            if shape[1].is_dynamic:
                                shape[1] = 0
                            inputs[input_name] = Tensor(model_inputs.get_element_type(),
                                                    shape.get_shape())
                        elif self.model_family == "chatglm":
                            if shape[0].is_dynamic:
                                shape[0] = 0
                            if shape[1].is_dynamic:
                                shape[1] = shape_input_ids[0]
                            inputs[input_name] = ov.Tensor(model_inputs.get_element_type(), shape.get_shape())
                cur_input_len = len(inputs["input_ids"][0])
                if "attention_mask" in self.input_names and attention_mask is not None:
                    inputs["attention_mask"] = attention_mask
                
                self.request.start_async(inputs, share_inputs=True)
                self.request.wait()
                logits = self.request.get_tensor("logits").data
                past_key_values = tuple(
                    self.request.get_tensor(key).data for key in self.key_value_output_names)
                next_token_logits = logits[:, cur_input_len - 1, :]
                # pre-process distribution
                next_token_scores = process_logits(
                    len(input_ids[0]),
                    next_token_logits,
                    eos_token_id
                )
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
                        ) == max_generated_tokens or next_tokens == eos_token_id:
                    break
                else:
                    input_ids = np.concatenate((input_ids, [next_tokens]), axis=-1)
                    attention_mask = np.concatenate(
                        (attention_mask, [[1] * len(next_tokens)]), axis=-1)

            finally:
                end = time.perf_counter()
                latency.append(end - st)
                st = end

        return input_ids

    
    def _generate_sequence_chatglm_advanced(
        self,
        input_ids,
        attention_mask,
        max_generated_tokens=100,
        eos_token_id=2,
        top_k=20,
        top_p=0.7,
        temperature=1,
        do_sample=False,
        perf: Optional[dict]=None
    ):
        from utils import get_top_k_logits, process_logits, sample_next_tokens
        prompt_len = len(input_ids[0])

        position_ids = np.arange(0, input_ids.shape[1], dtype=np.int64)
        position_ids = np.expand_dims(position_ids, axis=0)
        past_key_values = None
        new_position_id = np.copy(position_ids[..., -1:])
        if perf is not None:
            latency = perf["latency"]
            st = time.perf_counter()
        while True:
            try:
                inputs = {}
                if past_key_values is not None:
                    inputs = dict(zip(self.key_value_input_names, past_key_values))
                    inputs["input_ids"] = input_ids[:, -1:]
                    new_position_id += 1
                    inputs["position_ids"] = new_position_id
                else:
                    inputs["input_ids"] = input_ids
                    inputs["position_ids"] = position_ids
                    shape_input_ids = input_ids.shape
                    for input_name in self.key_value_input_names:
                        model_inputs = self.model.input(input_name)
                        shape = model_inputs.get_partial_shape()
                        if shape[0].is_dynamic:
                            shape[0] = 0
                        if shape[1].is_dynamic:
                            shape[1] = shape_input_ids[0]
                        inputs[input_name] = ov.Tensor(
                            model_inputs.get_element_type(), shape.get_shape())
                    
                if "attention_mask" in self.input_names and attention_mask is not None:
                    inputs["attention_mask"] = attention_mask

                self.request.start_async(inputs, share_inputs=True)
                self.request.wait()

                # get next tokens
                logits = self.request.get_tensor("logits").data
                past_key_values = tuple(
                    self.request.get_tensor(key).data for key in self.key_value_output_names)
                if do_sample:
                    next_tokens = sample_next_tokens(logits=logits[0, -1],top_p=top_p, top_k=top_k, temperature=temperature)
                else:
                    next_token_logits = logits[:, -1, :] # diff from llama logits[:, cur_input_len - 1, :]
                    # pre-process distribution
                    next_token_scores = process_logits(
                        len(input_ids[0]),
                        next_token_logits,
                        eos_token_id
                    )
                    next_token_scores = get_top_k_logits(next_token_scores, top_k)
                    # next_token = np.argmax(next_token_scores, axis=-1)[0].item() # greedy search
                    next_tokens = np.argmax(next_token_scores, axis=-1)
                if (len(input_ids[0]) - prompt_len
                        ) == max_generated_tokens or (next_tokens == eos_token_id).any():
                    break
                else:
                    input_ids = np.concatenate((input_ids, [next_tokens]), axis=-1)
                    attention_mask = np.concatenate(
                        (attention_mask, [[1] * len(next_tokens)]), axis=-1)
            finally:
                if perf is not None:
                    end = time.perf_counter()
                    latency.append(end - st)
                    st = end
        return input_ids

    def _generate_sequence_chatglm_origin(
        self,
        input_ids,
        max_generated_tokens=128,
        eos_token_id=2,
        top_k=20,
        top_p=0.7,
        temperature=1,
        do_sample=False,
        perf: Optional[dict]=None
    ):
        from utils import get_top_k_logits, process_logits, sample_next_tokens
        
        if perf is None:
            perf = {"latency":[]}
        latency = perf["latency"]
        st = time.perf_counter()
        attention_mask = np.ones((input_ids.shape[0], input_ids.shape[1]),
                                 dtype=np.int64)
        position_ids = np.arange(0, input_ids.shape[1], dtype=np.int64)
        position_ids = np.expand_dims(position_ids, axis=0)
        past_key_values = None
        new_position_id = np.copy(position_ids[..., -1:])
        output_tokens = []
        while True:
            try:
                inputs = {"input_ids": input_ids}
                if past_key_values is not None:
                    new_position_id += 1
                    inputs["position_ids"] = new_position_id
                    inputs.update(past_key_values)
                else:
                    inputs["position_ids"] = position_ids
                    shape_input_ids = input_ids.shape
                    for input_name in self.key_value_input_names:
                        model_inputs = self.model.input(input_name)
                        shape = model_inputs.get_partial_shape()
                        if shape[0].is_dynamic:
                            shape[0] = 0
                        if shape[1].is_dynamic:
                            shape[1] = shape_input_ids[0]
                        inputs[input_name] = ov.Tensor(
                            model_inputs.get_element_type(), shape.get_shape())

                if "attention_mask" in self.input_names and attention_mask is not None:
                    inputs["attention_mask"] = attention_mask
                self.request.start_async(inputs, share_inputs=True)
                self.request.wait()
                logits = self.request.get_tensor("logits").data
                past_key_values = tuple(self.request.get_tensor(key).data for key in self.key_value_output_names)
                past_key_values = {k: v for k, v in zip(self.key_value_input_names, past_key_values)}
                if do_sample:
                    next_token = sample_next_tokens(logits=logits[0, -1],top_p=top_p, top_k=top_k, temperature=temperature)[0].item()
                else:
                    next_token_logits = logits[:, -1, :]
                    next_token_scores = process_logits(
                        len(input_ids[0]),
                        next_token_logits,
                        eos_token_id
                    )
                    next_token_scores = get_top_k_logits(next_token_scores, top_k)
                    next_token = np.argmax(next_token_scores, axis=-1)[0].item()
                output_tokens += [next_token]
                if next_token in self.eos_token_id or len(output_tokens) > max_generated_tokens:
                    break
                attention_mask = np.concatenate((attention_mask, [[1]]), axis=-1)
                input_ids = np.array([[next_token]], dtype=np.longlong)
            finally:
                end = time.perf_counter()
                latency.append(end - st)
                st = end
        return output_tokens
    
    def _warmup(self, prompt):
        log.info(f'Warm up')
        # inputs = self.tokenizer(prompt, return_tensors="np")
        from utils import build_inputs
        input_ids,inputs = build_inputs(tokenizer=self.tokenizer, query=prompt, history=[], return_tensors="np")
        attention_mask = inputs["attention_mask"]
        
        if self.model_family == "llama":
            self._generate_sequence_llama(
                sampling=False,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_sequence_length=50,
                eos_token_id=self.tokenizer.eos_token_id
            )
        elif self.model_family == "chatglm":
            self._generate_sequence_chatglm_advanced(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                eos_token_id=self.tokenizer.eos_token_id,
                max_generated_tokens=128,
                do_sample=False,
            )
            # self.eos_token_id = [self.tokenizer.eos_token_id]
            # self._generate_sequence_chatglm_origin(
            #         input_ids=input_ids,
            #         eos_token_id=self.tokenizer.eos_token_id,
            #         max_generated_tokens=128,
            #         do_sample=False,
            #     )

    def create_completion(self, prompt, max_new_tokens, top_p, temperature, num_beams, do_sample, stream=False):
        if not stream:
            perf = {"latency": [], }
            log.info(f'Start generating')
            st = time.perf_counter()
            from utils import build_inputs
            input_ids,inputs = build_inputs(tokenizer=self.tokenizer, query=prompt, history=[], return_tensors="np")
            attention_mask = inputs["attention_mask"]
            prompt_tokens = input_ids.shape[1]
            if self.model_family == "llama":
                output_ids = self._generate_sequence_llama(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_generated_tokens=max_new_tokens if max_new_tokens else 128,
                    top_p=top_p if top_p else 0.7,
                    temperature=temperature if temperature else 0.95,
                    eos_token_id=self.tokenizer.eos_token_id,
                    sampling=do_sample,
                    perf=perf
                )
                completion_ids = output_ids[0].tolist()[prompt_tokens:]
                completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            if self.model_family == "chatglm":
                output_ids = self._generate_sequence_chatglm_advanced(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_generated_tokens=max_new_tokens,
                    do_sample=do_sample,
                    perf=perf
                )
                prompt_tokens = input_ids.shape[1]
                completion_ids = output_ids[0].tolist()[prompt_tokens:]
                completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

                # self.eos_token_id = [self.tokenizer.eos_token_id]
                # completion_ids = self._generate_sequence_chatglm_origin(
                #     input_ids=input_ids,
                #     max_generated_tokens=max_new_tokens,
                #     top_p=top_p if top_p else 0.7,
                #     temperature=temperature if temperature else 0.95,
                #     perf=perf,
                #     do_sample=do_sample
                # )
                # completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

            end = time.perf_counter()
            latency = perf["latency"]
            resp = {
                "completion": completion,
                "prompt_tokens": prompt_tokens,
                "total_dur_s": end-st,  # total time, include tokeninzer.encode+decode, tokens generation
                "completion_tokens": len(completion_ids),
                # total tokens completion latency, except tokenizer.decode time
                "total_token_latency_s": sum(latency),
                # first token completion latency
                "first_token_latency_ms": latency[0]*1000 if len(latency) > 0 else 0,
                # next token completion latency
                "next_token_latency_ms": sum(latency[1:])*1000 / len(latency[1:]) if len(latency) > 1 else 0,
                # average token completion latency
                "avg_token_latency_ms": sum(latency)*1000 / len(latency) if len(latency) > 0 else 0,
            }
            return resp
        else:
            # TODO stream
            pass
        return
    
