
import argparse
import time
import psutil
import os
from threading import Event, Thread

import logging as log
import sys
log.basicConfig(format='[ %(levelname)s ] %(message)s',
                level=log.INFO, stream=sys.stdout)


class MemConsumption:
    def __init__(self):
        """Initialize MemConsumption."""
        self.g_exit_get_mem_thread = False
        self.g_end_collect_mem = False
        self.g_max_rss_mem_consumption = -1
        self.g_max_shared_mem_consumption = -1
        self.g_event = Event()
        self.g_data_event = Event()

    def collect_memory_consumption(self):
        """Collect the data."""
        while self.g_exit_get_mem_thread is False:
            self.g_event.wait()
            while True:
                process = psutil.Process(os.getpid())
                rss_mem_data = process.memory_info().rss / float(2**20)
                try:
                    shared_mem_data = process.memory_info().shared / float(2**20)
                except Exception:
                    shared_mem_data = -1
                if rss_mem_data > self.g_max_rss_mem_consumption:
                    self.g_max_rss_mem_consumption = rss_mem_data
                if shared_mem_data > self.g_max_shared_mem_consumption:
                    self.g_max_shared_mem_consumption = shared_mem_data
                self.g_data_event.set()
                if self.g_end_collect_mem is True:
                    self.g_event.set()
                    self.g_event.clear()
                    self.g_end_collect_mem = False
                    break
                time.sleep(500 / 1000)

    def start_collect_memory_consumption(self):
        """Start collect."""
        self.g_end_collect_mem = False
        self.g_event.set()

    def end_collect_momory_consumption(self):
        """Stop collect."""
        self.g_end_collect_mem = True
        self.g_event.wait()

    def get_max_memory_consumption(self):
        """Return the data."""
        self.g_data_event.wait()
        self.g_data_event.clear()
        return self.g_max_rss_mem_consumption, self.g_max_shared_mem_consumption

    def clear_max_memory_consumption(self):
        """Clear MemConsumption."""
        self.g_max_rss_mem_consumption = -1
        self.g_max_shared_mem_consumption = -1

    def start_collect_mem_consumption_thread(self):
        """Start the thread."""
        self.t_mem_thread = Thread(target=self.collect_memory_consumption)
        self.t_mem_thread.start()

    def end_collect_mem_consumption_thread(self):
        """End the thread."""
        self.g_event.set()
        self.g_data_event.set()
        self.g_end_collect_mem = True
        self.g_exit_get_mem_thread = True
        self.t_mem_thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h',
                        '--help',
                        action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-m',
                        '--model_path',
                        required=True,
                        type=str,
                        help='Required. model path')
    parser.add_argument('-p',
                        '--prompt',
                        default="what is ai?",
                        required=False,
                        type=str,
                        help='prompt')
    parser.add_argument('-l',
                        '--max_sequence_length',
                        default=128,
                        required=False,
                        type=int,
                        help='Required. maximun lengh of output')
    parser.add_argument('-d',
                        '--device',
                        default='CPU',
                        required=False,
                        type=str,
                        help='Required. device for inference')
    args = parser.parse_args()

    model_id = args.model_path
    if 'chatglm2' in model_id:
        from chatglm.modeling import ChatGLM2Model
        ov_model = ChatGLM2Model(model_id, args.device)
    elif 'chatglm3' in model_id:
        from chatglm.modeling import ChatGLM3Model
        ov_model = ChatGLM3Model(model_id, args.device)
    else:
        raise NotImplementedError(f"Unsupported model id {model_id!r}")

    mem_consumption = MemConsumption()
    max_rss_mem_consumption = ''
    max_shared_mem_consumption = ''
    mem_consumption.start_collect_mem_consumption_thread()
    mem_consumption.start_collect_memory_consumption()

    input_data = ov_model.build_inputs([], args.prompt)
    input_len = len(input_data[0])

    log.info(" --- start generating --- ")
    start = time.perf_counter()
    response, num_tokens, latencies = ov_model.generate_sequence(
        input_data, max_generated_tokens=args.max_sequence_length)
    end = time.perf_counter()
    output_data = ov_model.tokenizer.decode(response, skip_special_tokens=True)
    answer, _ = ov_model.process_response(output_data, [])
    log.info(f"Response: {answer}")

    mem_consumption.end_collect_momory_consumption()
    max_rss_mem_consumption, max_shared_mem_consumption = mem_consumption.get_max_memory_consumption()
    mem_consumption.clear_max_memory_consumption()

    log.info(" --- Benchmarking --- ")
    log.info(f"Input length: {input_len} tokens")
    log.info(
        f"Generated {num_tokens} tokens in {end - start:.2f} s on {args.device}")
    log.info(
        f"Maximum rss memory consumption: {max_rss_mem_consumption:.2f} MB, Maximum shared memory consumption: {max_shared_mem_consumption:.2f}  MB")
    log.info(
        f"First inference latency: {1000*latencies[0]:.2f} ms/token, Other inference latency {1000*latencies[1]/(num_tokens-1):.2f} ms/token in average")
    mem_consumption.end_collect_mem_consumption_thread()
