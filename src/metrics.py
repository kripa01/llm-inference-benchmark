import time
import psutil
import os


def measure_time(func):
    start = time.perf_counter()

    result = func()

    end = time.perf_counter()

    return result, end - start


def get_memory_usage():
    process = psutil.Process(os.getpid())

    memory_mb = process.memory_info().rss / (1024 ** 2)

    return round(memory_mb, 2)


def calculate_tokens_per_second(tokens, seconds):
    if seconds == 0:
        return 0

    return round(tokens / seconds, 2)