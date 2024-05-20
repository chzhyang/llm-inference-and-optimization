import subprocess
import os
# import signal
# import re
# import asyncio
import grpc
import xft_pb2
import xft_pb2_grpc
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc
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
def stop_service(addr) -> None:
    with grpc.insecure_channel(addr) as channel:
        stub=xft_pb2_grpc.XFTServiceStub(channel)
        health_stub = health_pb2_grpc.HealthStub(channel)
        if not health_check_call(health_stub):
            print(f"[ERROR] XFT server is not ready on {addr}")
        try: 
            print("[client]Sending stop request to server", flush=True)
            response =  stub.stop_service(
                xft_pb2.StopServiceRequest(stop_service=True)
            )
            print("[client]",response.message, flush=True)
        except Exception:
            pass
# async def stop_service(addr) -> None:
#     # with grpc.insecure_channel(addr) as channel:
#     async with grpc.aio.insecure_channel(addr) as channel:
#         stub=xft_pb2_grpc.XFTServiceStub(channel)
#         health_stub = health_pb2_grpc.HealthStub(channel)
#         if not health_check_call(health_stub):
#             print(f"[ERROR] XFT server is not ready on localhost:{args.ip}:{args.port}")
#         try: 
#             print("[client]Sending stop request to server", flush=True)
#             response = await stub.stop_service(
#                 xft_pb2.StopServiceRequest(stop_service=True)
#             )
#             print("[client]",response.message, flush=True)
#         except Exception:
#             pass
        
# 执行mpirun命令启动两个进程
cmd="OMP_NUM_THREADS=60 mpirun -n 1 numactl -C 0-59 -m 0 python server.py : -n 1 numactl  -C 60-119 -m 1 python server.py"
process = subprocess.Popen(cmd, preexec_fn=os.setsid, shell=True)
print("[main]sleep 30s", flush=True)
import time
time.sleep(30)
print("[main]sending stop request", flush=True)
# from client import stop_service
# asyncio.run(stop_service(f"localhost:50051"))
stop_service(f"localhost:50051")
print("[main]Waiting for process", flush=True)
process.wait()
print("[main]process terminated", flush=True)

# print("create new mpi work", flush=True)
# process = subprocess.Popen(cmd, preexec_fn=os.setsid, shell=True)
# print("sleep 30s", flush=True)
# import time
# time.sleep(30)
# print("sending stop request", flush=True)
# from client import stop_service
# stop_service(f"localhost:50051")
# print("等待进程完成", flush=True)
# process.wait()
# print("process terminated", flush=True)
# print("sleep 60s")
# import time
# time.sleep(60)
# print("停止进程，发送SIGTERM信号")
# # # 停止进程，发送SIGTERM信号
# # process.terminate()
# os.killpg(os.getpgid(process.pid), signal.SIGTERM)

# print("等待子进程终止")
# # 等待子进程终止
# process.wait()
# print("子进程终止")


# pid_pattern = re.compile(r'\d+\s+(\d+)\s+\d+')
# str=f'''
# [0] MPI startup(): Intel(R) MPI Library, Version 2021.10  Build 20230605 (id: 32f12718fe)
# [0] MPI startup(): Copyright (C) 2003-2023 Intel Corporation.  All rights reserved.
# [0] MPI startup(): library kind: release
# [0] MPI startup(): libfabric version: 1.16.1-ccl
# [0] MPI startup(): libfabric provider: tcp;ofi_rxm
# [0] MPI startup(): File "/app/xft/3rdparty/oneccl/build/_install/etc/tuning_spr_shm-ofi_tcp-ofi-rxm_10.dat" not found
# [0] MPI startup(): Load tuning file: "/app/xft/3rdparty/oneccl/build/_install/etc/tuning_spr_shm-ofi.dat"
# [0] MPI startup(): File "/app/xft/3rdparty/oneccl/build/_install/etc/tuning_spr_shm-ofi.dat" not found
# [0] MPI startup(): Load tuning file: "/app/xft/3rdparty/oneccl/build/_install/etc//tuning_clx-ap_shm-ofi.dat"
# [0] MPI startup(): THREAD_SPLIT mode is switched on, 1 endpoints in use
# [0] MPI startup(): Rank    Pid      Node name     Pin cpu
# [0] MPI startup(): 0       157472   7e4ef06c811b  0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,
#                                     30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56
#                                     ,57,58,59
# [0] MPI startup(): 1       157473   7e4ef06c811b  60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86
#                                     ,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,
#                                     110,111,112,113,114,115,116,117,118,119'''
# print("test获取pids")
# pids = pid_pattern.findall(str)

# print("获取到的PID:", pids)

# print("获取pids")
# stdout, stderr = process.communicate()
# pids = pid_pattern.findall(stdout.decode())
# print("获取到的PID:", pids, flush=True)