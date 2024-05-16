from contextlib import contextmanager
from threading import Thread
import sys

from queue import Queue
from typing import List
import torch


class Pack:
    def __init__(self, *args):
        self.args = args
        
    def call(self, func):
        return Pack(func(*self.args))
    

class Task:
    def __init__(self, compute_fn):
        self.compute_fn = compute_fn
    
    def compute(self):
        return self.compute_fn()
    

def worker(in_queue: Queue, out_queue: Queue):
    while True:
        task = in_queue.get()
        if task is None:
            break
        
        try:
            pack = task.compute()
            out_queue.put((True, pack))
        except Exception:
            exc_info = sys.exc_info()
            out_queue.put((False, exc_info))
            
    out_queue.put((False, None))
    

@contextmanager
def spawn_device_workers(num_stages):
    in_queues = []
    out_queues = []
    
    for _ in range(num_stages):
        in_queue = Queue()
        out_queue = Queue()
        
        t = Thread(target=worker, args=(in_queue, out_queue), daemon=True)
        t.start()

        in_queues.append(in_queue)
        out_queues.append(out_queue)
        
    try:
        yield (in_queues, out_queues)
    finally:
        for in_queue in in_queues:
            in_queue.put(None)
        
        # Join running threads
        running = set(out_queues)
        while running:
            out_queue = running.pop()
            ok, info = out_queue.get()
            if info is None:
                continue
            
            running.add(out_queue)
