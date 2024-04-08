import psutil
import os


def get_free_ram():
    mem = psutil.virtual_memory()
    return mem.available / (1024 * 1024 * 1024)


def get_total_ram():
    mem = psutil.virtual_memory()
    return mem.total / (1024 * 1024 * 1024)


def get_ram_in_use():
    mem = psutil.virtual_memory()
    return mem.used / (1024 * 1024 * 1024)


def get_current_process_ram_usage() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024 * 1024)


def ram_load_check(load: float = 0.80) -> bool:
    return not get_ram_in_use()/get_total_ram() >= load

