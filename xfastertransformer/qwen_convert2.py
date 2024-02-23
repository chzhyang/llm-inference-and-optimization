import xfastertransformer as xft
HF_DATASET_DIR="/home/ge/models/Qwen-7B-Chat"
OUTPUT_DIR="/home/ge/models/qwen-7b-chat-xft"
# Qwen don't support convert weight to fp16, since there is some known issues. Please use fp32 data type.
xft.QwenConvert().convert(HF_DATASET_DIR,OUTPUT_DIR, dtype="fp32", processes=60)