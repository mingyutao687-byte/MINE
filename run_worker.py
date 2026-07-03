#!/usr/bin/env python
"""vLLM Worker 启动器。

用法:
  # 在所有 GPU 上各启动 1 个 Worker (一键)
  python run_worker.py --gpu-count 4 --model /path/to/model

  # 指定单 GPU
  python run_worker.py --gpu 0 --port 8000 --model /path/to/model

  # CPU Worker
  python run_worker.py --device cpu --port 8400 --model /path/to/model --numa 0 --kv_gb 32
"""

import sys
import os
import argparse
import subprocess
import shlex
import time
import signal

_MINE_DIR = os.path.dirname(os.path.abspath(__file__))
_MINE_PARENT = os.path.dirname(_MINE_DIR)

BASE_ENV = {
    **os.environ,
    'PYTHONPATH': os.pathsep.join([
        _MINE_PARENT,
        os.environ.get('PYTHONPATH', ''),
    ]),
    'VLLM_ENGINE_ITERATION_TIMEOUT_S': '3600',
}


def start_gpu_worker(gpu_id: int, port: int, model: str, block_num: int,
                     max_model_len: int) -> subprocess.Popen:
    env = {**BASE_ENV, 'CUDA_VISIBLE_DEVICES': str(gpu_id),
           'NO_MODEL_LOADING_AT_START': 'True'}
    block_override = f'--num-gpu-blocks-override {block_num}' if block_num >= 0 else ''
    cmd = (f'python -m vllm.entrypoints.openai.api_server '
           f'--port {port} --max-model-len {max_model_len} '
           f'--model {model} --enforce-eager '
           f'--gpu-memory-utilization 0.975 --block-size 16 '
           f'{block_override}')
    print(f'  [GPU {gpu_id}] :{port}  {model}')
    return subprocess.Popen(shlex.split(cmd), env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start vLLM worker(s)')
    parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])
    parser.add_argument('--gpu-count', type=int, default=None,
                       help='Number of GPUs — starts one worker per GPU')
    parser.add_argument('--port', type=int, default=None,
                       help='Port (auto-assigned if --gpu-count used)')
    parser.add_argument('--base-port', type=int, default=8000,
                       help='Starting port when using --gpu-count')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=None,
                       help='Single GPU ID (ignored if --gpu-count used)')
    parser.add_argument('--numa', type=int, default=None)
    parser.add_argument('--kv_gb', type=int, default=None)
    parser.add_argument('--block_num', type=int, default=256)
    parser.add_argument('--max_model_len', type=int, default=4096)
    args = parser.parse_args()

    # --- GPU count mode: start workers on all GPUs ---
    if args.gpu_count:
        procs = []
        for gpu_id in range(args.gpu_count):
            port = args.base_port + gpu_id
            proc = start_gpu_worker(gpu_id, port, args.model,
                                    args.block_num, args.max_model_len)
            procs.append(proc)
        print(f'\nStarted {len(procs)} GPU workers. Ctrl+C to stop all.')
        try:
            for p in procs:
                p.wait()
        except KeyboardInterrupt:
            print('\nStopping all...')
            for p in procs:
                p.kill()
            for p in procs:
                p.wait()
            print('All stopped.')
        sys.exit(0)

    # --- Single worker mode ---
    if args.device == 'gpu':
        assert args.gpu is not None, '--gpu or --gpu-count required'
        assert args.port is not None, '--port required'
        proc = start_gpu_worker(args.gpu, args.port, args.model,
                                args.block_num, args.max_model_len)
    elif args.device == 'cpu':
        assert args.port is not None, '--port required'
        assert args.kv_gb is not None, '--kv_gb required'
        env = {**BASE_ENV, 'VLLM_OPENVINO_KVCACHE_SPACE': str(args.kv_gb)}
        numa_cmd = ''
        if args.numa is not None and args.numa >= 0:
            numa_cmd = f'numactl --cpunodebind={args.numa} --membind={args.numa}'
        cmd = (f'{numa_cmd} python -m vllm.entrypoints.openai.api_server '
               f'--port {args.port} --max-model-len {args.max_model_len} '
               f'--model {args.model}')
        print(f'  [CPU] :{args.port}  {args.model}')
        proc = subprocess.Popen(shlex.split(cmd), env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
