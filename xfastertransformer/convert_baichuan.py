import xfastertransformer as xft

xft.BaichuanConvert().split_and_convert(input_dir="/home/ge/models/Baichuan2-13B-Base", output_dir="/home/ge/models/baichuan2-13b-xft-fp16", dtype="fp16", processes=60)