import re
import subprocess
# 在main.py中执行mpirun -n2 python inference.py

model_path="/home/ge/models/qwen-7b-chat-xft/"
token_path="/home/ge/models/Qwen-7B-Chat"
dtype="bf16_fp16"
prompt="What is ai?"
output_len=10
# generate=f"python generate.py --model_path={model_path} --token_path={token_path} --dtype={dtype}"
generate=f"./example --model {model_path} --token {token_path} --dtype {dtype} --input {prompt} --output_len {str(output_len)} --no_stream --warmup"
def single2(generate):
    n_thread_per_process=60
    cmd = f'OMP_NUM_THREADS={str(n_thread_per_process)} numactl -m 0 -C 0-59 {generate}'
    result = ""
    try:
        process = subprocess.Popen([cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                result += output
                print(output.strip())
        process.wait()
        # return_code = process.returncode
        # error_output = process.stderr.read()
        # print(return_code, error_output)
        match(result)
    except subprocess.CalledProcessError as e:
        print(str(e.output))

def single(generate):
  n_thread_per_process=60
  cmd = f'OMP_NUM_THREADS={str(n_thread_per_process)} numactl -m 0 -C 0-59 {generate}'
  try:
    output = subprocess.run([cmd], capture_output=True, shell=True, text=True)
    # process = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, text=True)
    print("Output:\n", output.stdout)
    # Parse output
    match(output.stdout)
  except subprocess.CalledProcessError as e:
    print(str(e.output))

def match(result):
    prompt_len_match = re.search(r'Prompt length is: (\d+)\n',result)
    if prompt_len_match:
        prompt_tokens = int(prompt_len_match.group(1))
    generation_len_match = re.search(r'Generation length is: (\d+)\n',result)
    if generation_len_match:
      generation_tokens = int(generation_len_match.group(1))
    first_token_time_match = re.search(r'First token time: (\d+\.\d+) ms',result)
    if first_token_time_match:
      first_token_latency = float(first_token_time_match.group(1))
    next_token_time_match = re.search(r'Next token time: (\d+\.\d+) ms',result)
    if next_token_time_match:
      next_token_latency = float(next_token_time_match.group(1))
    generation_match = re.search(r'Final output is: (.+)', result, re.DOTALL)
    if generation_match:
      generation = generation_match.group(1).strip()
    print(prompt_tokens)
    print(generation_tokens)
    print(first_token_latency)
    print(next_token_latency)
    print(generation)

def multi():
  n_thread_per_process=30
  n_process=4 # 1 host, 2的倍数
  n_socket=2 #numa节点数=n个mpi组，每组若干mpi进程
  process_per_socket=n_process/n_socket
  # 
  # 4个mpi进程，2组，每组分配一个numa节点，每个进程线程数为30
  cmd = f'OMP_NUM_THREADS={str(n_thread_per_process)} mpirun '

  for i in range(n_socket):
    mpi_cmd = f'-n {str(process_per_socket)} numactl -N {str(i)} -m {str(i)} {generate}'
  # + generate + " : -n {str(process_per_socket)} numactl -N 1 -m 1 " + generate'

  # cmd = f'OMP_NUM_THREADS=30 LD_PRELOAD=libiomp5.so mpirun -n 2 numactl -N 0 -m 0 python generate.py --model_path=/models/llama2-7b-xft --token_path=/models/llama2-7b-xft  --dtype=bf16 : numactl -N 1 -m 1 python generate.py --model_path=/models/llama2-7b-xft --token_path=/models/llama2-7b-xft --dtype=bf16'
  try:
      output = subprocess.check_output(cmd, shell=True, text=True)
      # process = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, text=True)
      print("Output from generate:", output)
  except subprocess.CalledProcessError as e:
      output = str(e.output)

mystr="""
INFO] Load tokenizer
[INFO] Load model
[INFO] SINGLE_INSTANCE MODE.
[INFO] Warmup
[INFO] Prompt length is: 16
[INFO] Start to generate
[INFO] First token time: 69.959 ms
[INFO] Next token time: 67.4009 ms
[INFO] Generation length is: 99
[INFO] Final output is: Once upon a time, there existed a little girl who liked to have adventures. She would often wander off into the woods, exploring and discovering new things. One day, she stumbled upon a hidden cave. It was dark and mysterious, and she could hear strange noises coming from inside. The little girl was curious and decided to explore the cave.\n\nAs she made her way deeper into the cave, she noticed that the walls were covered in strange symbols and markings. She couldn't make out what they meant, but she knew that they were important. Suddenly, she heard a loud rum
"""

if __name__ == "__main__":
    single2(generate)
    # match(result=mystr)