model1=/home/ge/models/Llama-2-7b-hf
model2=/models/Mixtral-8x7B-Instruct-v0.1.safetensors
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run
model_hub=/home/ge/models
numactl -m 0 -C 0-59 docker run --shm-size 4g -p 8080:80 -v $model_hub:/models  --cap-add=sys_nice --ipc=host ghcr.io/huggingface/tgi:1.4.1 --model-id $model2
