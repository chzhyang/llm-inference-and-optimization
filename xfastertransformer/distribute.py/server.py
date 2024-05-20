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
import asyncio

from concurrent import futures

import grpc
from xft_pb2 import GenerateRequest,GenerateResponse,StopServiceRequest,StopServiceResponse
from xft_pb2_grpc import XFTService,add_XFTServiceServicer_to_server

from grpc_health.v1 import health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

import os
import xfastertransformer
import torch
# Ignore Tensor-RT warning from huggingface
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return low_string == "true"


DTYPE_LIST = [
    "fp16",
    "bf16",
    "int8",
    "w8a8",
    "int4",
    "nf4",
    "bf16_fp16",
    "bf16_int8",
    "bf16_w8a8",
    "bf16_int4",
    "bf16_nf4",
    "w8a8_int8",
    "w8a8_int4",
    "w8a8_nf4",
]

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_path", type=str, default="/app/llama2-7b-xft", help="Path to model file")
parser.add_argument("-d", "--dtype", type=str, choices=DTYPE_LIST, default="fp16", help="Data type")
parser.add_argument("--num_beams", help="Num of beams, default to 1 which is greedy search.", type=int, default=1)
parser.add_argument("--output_len", help="max tokens can generate excluded input.", type=int, default=100)
parser.add_argument("--do_sample", help="Enable sampling search, Default to False.", type=boolean_string, default=False)
parser.add_argument("--temperature", help="value used to modulate next token probabilities.", type=float, default=1.0)
parser.add_argument("--top_p", help="retain minimal tokens above topP threshold.", type=float, default=1.0)
parser.add_argument("--top_k", help="num of highest probability tokens to keep for generation", type=int, default=50)
parser.add_argument("--rep_penalty", help="param for repetition penalty. 1.0 means no penalty", type=float, default=1.0)
parser.add_argument("--port", help="serve port, default 50051.", type=int, default=50051)

args = parser.parse_args()


class XFTServerService(XFTService):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.stop_words_ids = None
        self.server = None
        self.active_calls = asyncio.Semaphore(0)
        # Stop words ids for QWen model
        if "qwen" in args.model_path.lower():
            if "chat" in args.model_path.lower():
                self.stop_words_ids = [[151645], [151644]]
            else:
                self.stop_words_ids = [[33975, 25], [151643]]

    async def predict(self, request: GenerateRequest, context: grpc.aio.ServicerContext) -> GenerateResponse:
        input_ids = torch.Tensor(request.Ids).to(torch.int64).view(request.batch_size, request.seq_len)
        generated_ids = self.model.generate(
            input_ids,
            max_length=input_ids.shape[-1] + args.output_len,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
            stop_words_ids=self.stop_words_ids,
        )

        return GenerateResponse(
            Ids=generated_ids.view(-1).tolist(), batch_size=generated_ids.shape[0], seq_len=generated_ids.shape[1]
        )

    async def predict_stream(self, request: GenerateRequest, context: grpc.aio.ServicerContext) -> GenerateResponse:
        input_ids = torch.Tensor(request.Ids).to(torch.int64).view(request.batch_size, request.seq_len)
        self.model.config(
            input_ids.shape[-1] + args.output_len,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
            stop_words_ids=self.stop_words_ids,
        )
        self.model.input(input_ids)

        while not model.is_done():
            next_token_id = model.forward()
            next_token_id = next_token_id.view(-1).tolist()
            yield GenerateResponse(Ids=next_token_id, batch_size=len(next_token_id), seq_len=1)
    
    async def start_service(self, listen_addr):
        add_XFTServiceServicer_to_server(self, self.server)
        self.server.add_insecure_port(listen_addr)
        # Set server health to `serving`
        health_servicer = health.HealthServicer(
            experimental_non_blocking=True,
            experimental_thread_pool=futures.ThreadPoolExecutor(max_workers=10),
        )
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, self.server)
        health_servicer.set("xft.Service", health_pb2.HealthCheckResponse.SERVING)

        await self.server.start()
        print(f"XFT service started, listening on {listen_addr}", flush=True)
        # await self.server.wait_for_termination()
    
    async def stop_service(self, request: StopServiceRequest, context: grpc.aio.ServicerContext) -> StopServiceResponse:
        if request.stop_service:
            try:
                print("XFT service got a stop request", flush=True)
                await self.server.stop(0)
            finally:
                print("XFT service stopped", flush=True)
                return StopServiceResponse(message="XFT service stopped")
            # except Exception as e:
            #     print("Error occurred while stopping server:", e, flush=True)
            #     return StopServiceResponse(message="Failed to stop service")

async def serve(model, listen_addr) -> grpc.aio.Server:
    xft_service = XFTServerService(model)
    xft_service.server = grpc.aio.server()
    await xft_service.start_service(listen_addr)
    return xft_service.server


if __name__ == '__main__':
    print("loading model")
    model = xfastertransformer.AutoModel.from_pretrained(args.model_path, dtype=args.dtype)
    print("rank-",model.rank,"loaded model",flush=True)
    if model.rank != 0:
        while True:
            model.generate()
    listen_addr = f"[::]:{args.port}"
    async def main() -> None:
        await (await serve(model, listen_addr)).wait_for_termination()
    asyncio.run(main())