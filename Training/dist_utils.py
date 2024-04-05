import torch.multiprocessing as mp
import torch.distributed as dist


def spawn_processes(fn, args):
    mp.spawn(fn, args=(*args,), nprocs=args[-1], join=True)


def cleanup():
    dist.destroy_process_group()