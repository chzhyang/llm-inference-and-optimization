# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import time
import asyncio
import grpc
import xft_pb2
import xft_pb2_grpc
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

from transformers import AutoTokenizer
import torch

import argparse


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return low_string == "true"


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--token_path", type=str, default="/app/llama2-7b-xft", help="Path to token file")
parser.add_argument("--ip", help="serve ip, default localhost.", type=str, default="localhost")
parser.add_argument("--port", help="serve port, default 50051.", type=int, default=50051)
parser.add_argument("-f", "--function", help="test function, default generate().", type=str, default="generate")
parser.add_argument("--output_len", help="max tokens can generate excluded input.", type=int, default=100)
parser.add_argument("--num_beams", help="Num of beams, default to 1 which is greedy search.", type=int, default=1)
parser.add_argument("--do_sample", help="Enable sampling search, Default to False.", type=boolean_string, default=False)
parser.add_argument("--temperature", help="value used to modulate next token probabilities.", type=float, default=1.0)
parser.add_argument("--top_p", help="retain minimal tokens above topP threshold.", type=float, default=1.0)
parser.add_argument("--top_k", help="num of highest probability tokens to keep for generation", type=int, default=50)
parser.add_argument("--rep_penalty", help="param for repetition penalty. 1.0 means no penalty", type=float, default=1.0)

args = parser.parse_args()


def health_check_call(stub: health_pb2_grpc.HealthStub):
    start_time = time.time()
    request = health_pb2.HealthCheckRequest(service="xft.Service")
    while True:
        try:
            resp = stub.Check(request)
            if resp.status == health_pb2.HealthCheckResponse.SERVING:
                return True
            elif resp.status == health_pb2.HealthCheckResponse.NOT_SERVING:
                return False
        except grpc._channel._InactiveRpcError as e:
            pass

        elapsed_time = time.time() - start_time
        if elapsed_time >= 30:
            print("Health check timed out.")
            return False

        time.sleep(1)

# def generate():
#     tokenizer = AutoTokenizer.from_pretrained(args.token_path, use_fast=False, padding_side="left", trust_remote_code=True)
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token_id = tokenizer.eos_token_id

#     prompts = [
#         "Once upon a time, there existed a little girl who liked to have adventures.",
#         "Intel Corporation (commonly known as Intel) is an American multinational corporation and technology company.",
#     ]

#     TokenIds = tokenizer(prompts, return_tensors="pt", padding=True).input_ids

#     with grpc.insecure_channel(f"{args.ip}:{args.port}") as channel:
#         stub = xft_pb2_grpc.XFTServiceStub(channel)
#         health_stub = health_pb2_grpc.HealthStub(channel)
#         if not health_check_call(health_stub):
#             print(f"[ERROR] XFT server is not ready on {args.ip}:{args.port}")

#         response = stub.predict(
#             xft_pb2.GenerateRequest(
#                 Ids=TokenIds.view(-1).tolist(),
#                 batch_size=TokenIds.shape[0],
#                 seq_len=TokenIds.shape[-1],
#                 output_len = args.output_len,
#                 num_beams = args.num_beams,
#                 do_sample = args.do_sample,
#                 temperature = args.temperature,
#                 top_p = args.top_p,
#                 top_k = args.top_k,
#                 rep_penalty = args.rep_penalty,
#             )
#         )

#         response_ids = torch.Tensor(response.Ids).view(response.batch_size, response.seq_len)
#         ret = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

#         for snt in ret:
#             print(snt)

async def stop_service(addr) -> None:
    # with grpc.insecure_channel(addr) as channel:
    async with grpc.aio.insecure_channel(addr) as channel:
        stub=xft_pb2_grpc.XFTServiceStub(channel)
        # health_stub = health_pb2_grpc.HealthStub(channel)
        # if not health_check_call(health_stub):
        #     print(f"[ERROR] XFT server is not ready on localhost:{args.ip}:{args.port}")
        try: 
            print("[client]Sending stop request to server", flush=True)
            response = await stub.stop_service(
                xft_pb2.StopServiceRequest(stop_service=True)
            )
            print("[client]",response.message, flush=True)
        except Exception:
            pass
            # print("[client]",response.message, flush=True)
            
if args.function == "generate":
    # generate()
    pass
if args.function == "stop":
    asyncio.run(stop_service(f"{args.ip}:{args.port}"))
