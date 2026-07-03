"""vLLM 批量启动器 — 同时启动多个 vLLM Worker 进程。

用于一键启动一个节点上的所有 Worker 实例。
每个 Worker 输出重定向到 output/ 目录下的按时间戳命名的日志文件。
"""

import argparse
import datetime
import sys
import os
import subprocess
import shlex

# 获取 config_template 目录的路径以读取模型路径配置
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
_config_template_dir = os.path.join(_parent_dir, "config_template")
sys.path.insert(0, _config_template_dir)

from models_path_template import models_path_template  # type: ignore

# 每种模型的启动命令模板
MODELS_COMMAND_TEMPLATES = {
    'llama-2-13b': {
        'gpu': (
            f'python -m Mine.cli.backend_starter '
            f'--device gpu --block_num 256 --use_sllm True --no_load True '
            f'--model {models_path_template["llama-2-13b"]["gpu"]}'
        ),
        'cpu': (
            f'python -m Mine.cli.backend_starter '
            f'--device cpu --numa -1 '
            f'--model {models_path_template["llama-2-13b"]["cpu"]}'
        ),
    },
    'llama-2-7b': {
        'gpu': (
            f'python -m Mine.cli.backend_starter '
            f'--device gpu --block_num 256 --use_sllm True --no_load True '
            f'--model {models_path_template["llama-2-7b"]["gpu"]}'
        ),
        'cpu': (
            f'python -m Mine.cli.backend_starter '
            f'--device cpu --numa -1 '
            f'--model {models_path_template["llama-2-7b"]["cpu"]}'
        ),
    },
    'llama-3.2-3b': {
        'gpu': (
            f'python -m Mine.cli.backend_starter '
            f'--device gpu --block_num 256 --use_sllm False --no_load True '
            f'--model {models_path_template["llama-3.2-3b"]["gpu"]}'
        ),
        'cpu': (
            f'python -m Mine.cli.backend_starter '
            f'--device cpu --numa -1 '
            f'--model {models_path_template["llama-3.2-3b"]["cpu"]}'
        ),
    },
}


def _build_commands(args) -> list[tuple[str, str]]:
    """为每个 Worker 构建启动命令和输出文件名。"""
    nowtime = str(datetime.datetime.now())
    nowtime = nowtime.replace(':', '-').replace(' ', '-')

    if not os.path.exists('output'):
        os.mkdir('output')

    final_commands = []
    for worker_id in range(args.worker_num):
        cur_port = args.port + worker_id
        if args.device == 'gpu':
            assert args.gpu is not None
            cur_command = (
                f'{MODELS_COMMAND_TEMPLATES[args.model][args.device]} '
                f'--gpu {args.gpu} --port {cur_port}'
            )
            cur_output_file = f'{nowtime}_{args.model}_{args.gpu}_{cur_port}.out'
        elif args.device == 'cpu':
            assert args.cpu_kv_gb is not None
            cur_command = (
                f'{MODELS_COMMAND_TEMPLATES[args.model][args.device]} '
                f'--kv_gb {args.cpu_kv_gb} --port {cur_port}'
            )
            cur_output_file = f'{nowtime}_{args.model}_{cur_port}.out'
        else:
            raise ValueError(f'Invalid device: {args.device}')

        final_commands.append((cur_command, cur_output_file))

    return final_commands


def main():
    parser = argparse.ArgumentParser(description='Start multiple vLLM backend instances.')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--device', type=str, required=True, choices=['cpu', 'gpu'])
    parser.add_argument('--gpu', type=str, required=False, help='GPU device ID')
    parser.add_argument('--cpu_kv_gb', type=int, required=False, help='CPU KV cache GB')
    parser.add_argument('--port', type=int, required=True, help='Base port')
    parser.add_argument('--worker_num', type=int, required=True, help='Number of workers')

    args = parser.parse_args()

    if args.device == 'gpu' and args.gpu is None:
        raise ValueError('Argument gpu is needed for GPU mode')

    final_commands = _build_commands(args)

    processes = []
    for cmd, output_file in final_commands:
        print(f'cur_cmd: {cmd}, cur_output_file: {output_file}')
        with open(f'output/{output_file}', 'w') as f:
            process = subprocess.Popen(
                shlex.split(cmd),
                stdout=f,
                stderr=subprocess.STDOUT,
            )
            processes.append(process)

    try:
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        print('Main process received Ctrl-C.')
        for process in processes:
            process.kill()
        for process in processes:
            process.wait()


if __name__ == "__main__":
    main()
