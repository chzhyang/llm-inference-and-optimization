model1=models/Llama-2-7b-hf
model2=models/merged_medical_llama2_7b
model=$model1
volume=/home/ge/models # share a volume with the Docker container to avoid downloading weights every run

docker run --rm \
-e http_proxy=http://child-prc.intel.com:912 \
-e https_proxy=http://child-prc.intel.com:912 \
-e http_proxy=localhost,*.intel.com,0.0.0.0 \
--shm-size 1g \
-p 8080:80 \
-v $volume:/models \
ghcr.io/huggingface/text-generation-inference:2.0 \
--model-id $model \
--tokenizer-config-path ${model}/tokenizer_config.json \
--disable-custom-kernels
# docker run --rm --shm-size 1g -p 8080:80 -v $volume:/models -it ghcr.io/huggingface/text-generation-inference:2.0 --help[]