# Weights Compression using Optimum Intel(nncf)
from optimum.intel.openvino import OVModelForCausalLM
from optimum.intel import OVQuantizer
from transformers import LlamaTokenizer
import argparse
from pathlib import Path
import gc
if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h',
                        '--help',
                        action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-m',
                        '--model_id',
                        required=True,
                        type=str,
                        help='Required. hugging face model id')
    parser.add_argument('-o',
                        '--output',
                        required=False,
                        default="./ir_model",
                        type=str,
                        help='Required. path to save the ir model')
    parser.add_argument('-cw',
                        '--compress_weight',
                        required=False,
                        default=False,
                        type=bool,
                        help='compress model weight')

    args = parser.parse_args()

    model_path = Path(args.output)
    
    print(" --- exporting IR --- ")
    ov_model = OVModelForCausalLM.from_pretrained(args.model_id,
                                                  compile=False,
                                                  export=True)
    if args.compress_weight == False:
        # FP16
        ov_model.half()
        ov_model.save_pretrained(model_path)
    else:
        # INT8
        quantizer = OVQuantizer.from_pretrained(ov_model, task="text-generation")
        quantizer.quantize(save_directory=model_path, weights_only=True)
        del quantizer
    del ov_model
    gc.collect()
        
    print(" --- exporting tokenizer --- ")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_id)
    tokenizer.save_pretrained(model_path)