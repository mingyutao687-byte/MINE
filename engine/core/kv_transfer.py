"""KV Cache 传输层 — 基于 Gloo + TCP 的跨进程 KV tensor 传输。

这是 MINE 实现 CPU↔GPU 请求迁移的底层传输机制。

传输协议:
  1. Sender 通过 TCP socket 发送元数据 (pickle, ≤32KB):
     {request_id, tensor_shape, src_rank, dst_rank, dst_socket_addr, dst_socket_port, ...}
  2. Sender 通过 Gloo dist.send 发送 tensor 数据
  3. Receiver 先通过 TCP socket 接收元数据，分配 shared memory tensor
  4. Receiver 通过 Gloo dist.recv 接收 tensor 数据

为什么用 TCP + Gloo:
  - TCP: 交换元数据（tensor shape、request_id），实现握手
  - Gloo: PyTorch 的 CPU 分布式后端，支持高效的跨进程 tensor 传输

"""

import socket
import pickle
import threading
import time
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import psutil


# ============================================================================
# KV Sender — 发送线程
# ============================================================================

def kv_sender(send_queue: mp.Queue, logger_queue: mp.Queue):
    """KV cache 发送线程。

    持续从 send_queue 读取 (config, tensor) 对：
      1. 通过 TCP socket 发送 pickled config（元数据握手）
      2. 通过 Gloo dist.send 发送 tensor 数据
    """
    logger_queue.put('start to send kv')
    while True:
        tensor_config = send_queue.get()
        src_rank = tensor_config['src_rank']
        dst_rank = tensor_config['dst_rank']
        dst_sock_addr = tensor_config['dst_socket_addr']
        dst_sock_port = tensor_config['dst_socket_port']

        logger_queue.put(f'kv_tensor begin to send, info: {tensor_config}')

        tensor_data = send_queue.get()

        # 步骤 1: TCP 发送元数据
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((dst_sock_addr, dst_sock_port))
            tensor_config_in_bytes = pickle.dumps(tensor_config)
            assert len(tensor_config_in_bytes) < 32768  # 32KB 上限
            s.sendall(tensor_config_in_bytes)

        # 步骤 2: Gloo 发送 tensor
        dist.send(tensor=tensor_data, dst=dst_rank)


# ============================================================================
# KV Receiver — 接收线程
# ============================================================================

def kv_receiver(connection_config: dict, receive_queue: mp.Queue, logger_queue: mp.Queue):
    """KV cache 接收线程。

    1. 绑定 TCP socket，监听元数据
    2. 接收 pickled config，解析 tensor shape
    3. 分配 shared memory tensor
    4. 通过 Gloo dist.recv 接收 tensor 数据
    5. 将 (config, tensor) 放入 receive_queue 供主进程消费

    端口冲突处理: 先 kill 占用目标端口的进程
    """
    socket_addr = connection_config['socket_addr']
    socket_port = connection_config['socket_port']

    # 处理端口冲突
    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr.port == socket_port:
            pid = conn.pid
            if pid:
                logger_queue.put(f"Killing process with PID: {pid} on port {socket_port}")
                os.kill(pid, 9)
    time.sleep(1)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((socket_addr, socket_port))
        s.listen()
        logger_queue.put('start to receive kv')

        while True:
            conn, addr = s.accept()
            with conn:
                # 步骤 1: TCP 接收元数据
                data_config_bytes = conn.recv(32768)
            data_config = pickle.loads(data_config_bytes)

            src_rank = data_config['src_rank']
            tensor_shape = data_config['tensor_shape']
            logger_queue.put(f'kv_tensor begin to receive, info: {data_config}')

            # 步骤 2: 分配 shared memory tensor
            tensor = torch.empty(tensor_shape, dtype=torch.float16, device='cpu')
            tensor.share_memory_()

            # 步骤 3: Gloo 接收 tensor
            dist.recv(tensor=tensor, src=src_rank)

            # 步骤 4: 放入队列供主进程消费
            receive_queue.put(data_config)
            receive_queue.put(tensor)


# ============================================================================
# KV Transfer Manager — 传输进程入口
# ============================================================================

def kv_transfer_manager(
    connection_config: dict,
    send_queue: mp.Queue,
    receive_queue: mp.Queue,
    logger_queue: mp.Queue,
):
    """KV 传输子进程的入口函数。

    1. 初始化 Gloo 分布式进程组
    2. 启动 sender 线程 (kv_sender)
    3. 启动 receiver 线程 (kv_receiver)
    4. 等待两个线程完成（实际永不退出）

    Args:
        connection_config: {rank, world_size, master_addr, master_port, socket_addr, socket_port}
        send_queue: 发送队列 (从主进程接收待发送的 tensor)
        receive_queue: 接收队列 (向主进程交付收到的 tensor)
        logger_queue: 日志队列 (向主进程发送日志消息)
    """
    rank = connection_config['rank']
    world_size = connection_config['world_size']
    master_addr = connection_config['master_addr']
    master_port = connection_config['master_port']

    logger_queue.put(
        f'initializing gloo, rank: {rank}, world_size: {world_size}, '
        f'url: tcp://{master_addr}:{master_port}'
    )

    # 初始化 Gloo 分布式进程组
    dist.init_process_group(
        backend='gloo',
        rank=rank,
        world_size=world_size,
        init_method=f'tcp://{master_addr}:{master_port}',
    )

    # 启动 sender 和 receiver 线程
    t1 = threading.Thread(target=kv_sender, args=(send_queue, logger_queue))
    t2 = threading.Thread(target=kv_receiver, args=(connection_config, receive_queue, logger_queue))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
