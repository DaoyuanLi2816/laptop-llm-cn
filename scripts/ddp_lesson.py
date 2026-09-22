"""两进程 CPU/gloo 实验：证明 DDP 平均梯度与单进程大 batch 梯度相同。

不要求两张 GPU，也不是一套容错集群训练框架。每个 rank 使用等量有效 token；
如果 padding 数不同，先按有效 token 数归一化再谈等价，见系统章节。
"""

import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from laptop_llm.config import ModelConfig
from laptop_llm.model import LaptopLLM


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(123)
        cfg = ModelConfig(
            vocab_size=32, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=16
        )
        model = LaptopLLM(cfg)
        baseline = LaptopLLM(cfg)
        baseline.load_state_dict(model.state_dict())
        wrapped = DistributedDataParallel(model)
        ids = torch.arange(32).view(4, 8) % 32
        shard = ids[rank * 2 : (rank + 1) * 2]
        wrapped(shard, labels=shard).loss.backward()
        baseline(ids, labels=ids).loss.backward()
        for parameter, expected in zip(model.parameters(), baseline.parameters(), strict=True):
            torch.testing.assert_close(parameter.grad, expected.grad, atol=2e-6, rtol=1e-4)
        if rank == 0:
            print("PASS: 2-rank gloo gradients == single-process global batch gradients")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="laptop-llm-ddp-") as directory:
        rendezvous = (Path(directory) / "rendezvous").resolve().as_uri()
        mp.spawn(worker, args=(rendezvous,), nprocs=2, join=True)
