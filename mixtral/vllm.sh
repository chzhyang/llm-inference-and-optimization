docker run \
	--privileged \
	-it \
	--rm \
	--network=host \
	-v /home/ge/models:/models \
	-v /home/ge/llm-inference-and-optimization:/app/llm-inference-and-optimization \
	vllm-cpu-env \
	/bin/bash
