model=/models/Llama-2-70b-hf
volume=/.../models # share a volume with the Docker container to avoid downloading weights every run
port=9000
name=Llama-70b-gaudi
sudo docker run --rm --name $name -p ${port}:80 -v $volume:/models --runtime=habana -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host chzhyang/tgi-gaudi:2.0.0 --model-id $model --sharded true --num-shard 8 --max-input-length 1024 --max-total-tokens 2048
