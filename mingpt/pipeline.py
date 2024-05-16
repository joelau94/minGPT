from queue import Queue
import traceback
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Optional
from mingpt.worker import Pack, Task, worker, spawn_device_workers

# from torch.distributed.pipeline.sync.pipe import Pipe


class BlocksPipeline(nn.Module):

    def __init__(self, blocks: List[nn.Module], devices):
        super().__init__()
        self.blocks = blocks
        self.devices = devices

        self.num_blocks = len(blocks)
        self.num_devices = len(devices)

        assert self.num_blocks % self.num_devices == 0
        self.blocks_per_stage = self.num_blocks // self.num_devices
        assert self.blocks_per_stage > 2

        self.splitted_blocks = [
            blocks[i * self.blocks_per_stage:(i + 1) * self.blocks_per_stage]
            for i in range(self.num_devices)
        ]

        self.params_copy_streams = {
            device: torch.cuda.Stream(device)
            for device in devices
        }
        self.data_copy_streams = {
            device: torch.cuda.Stream(device)
            for device in devices
        }
        self.compute_streams = {
            device: torch.cuda.Stream(device)
            for device in devices
        }

        self._first_blocks_inited = False

    def init_first_blocks(self):
        if not self._first_blocks_inited:
            for device_id in range(self.num_devices):
                device = self.devices[device_id]
                block_id = self.blocks_per_stage * device_id
                with torch.cuda.stream(self.params_copy_streams[device]):
                    self.blocks[block_id] = self.blocks[block_id].to(
                        device, non_blocking=False)
            self._first_blocks_inited = True

    def forward(self, x: torch.Tensor):
        self.init_first_blocks()
        for block_id in range(self.num_blocks):
            local_block_id = block_id % self.blocks_per_stage
            device_id = block_id // self.blocks_per_stage
            device = self.devices[device_id]

            next_block_id = (device_id * self.blocks_per_stage +
                             (local_block_id + 1) % self.blocks_per_stage)

            params_copy_stream = self.params_copy_streams[device]
            compute_stream = self.compute_streams[device]

            data_move_event = None
            if device != x.device:
                with torch.cuda.stream(self.data_copy_streams[device]):
                    x = x.to(device)
                    data_move_event = torch.cuda.Event()
                    data_move_event.record()

            with torch.cuda.stream(compute_stream):
                if data_move_event is not None:
                    compute_stream.wait_event(data_move_event)
                x = self.blocks[block_id](x)
                compute_done_event = torch.cuda.Event()
                compute_done_event.record(compute_stream)

            with torch.cuda.stream(params_copy_stream):
                self.blocks[next_block_id] = self.blocks[next_block_id].to(
                    device, non_blocking=True)
                params_copy_stream.wait_event(compute_done_event)
                self.blocks[block_id] = self.blocks[block_id].to(
                    "cpu", non_blocking=True)

        return x


def create_gpu_devices(gpu_ids):
    return [torch.device(f"cuda:{i}") for i in gpu_ids]


class RotatingBlocksStage(nn.Module):

    def __init__(self, blocks: List[nn.Module], device, compute_stream=None, copy_stream=None):
        super().__init__()
        self.blocks = blocks
        self.device = device

        self.num_blocks = len(self.blocks)
        assert self.num_blocks > 2

        self.compute_stream = (compute_stream if compute_stream is not None else
                            torch.cuda.Stream(device))
        self.copy_stream = (copy_stream if copy_stream is not None else
                            torch.cuda.Stream(device))
        
        self.data_move_event = None
        self.compute_events = [None for _ in range(self.num_blocks)]
        self.param_load_events = [None for _ in range(self.num_blocks)]

        self._init_loaded = False

    def init_load(self):
        if not self._init_loaded:
            with torch.cuda.stream(self.copy_stream):
                self.blocks[0] = self.blocks[0].to(self.device,
                                                   non_blocking=False)
            self._init_loaded = True
            
    def recv_data(self, x):
        if self.device != x.device:
            with torch.cuda.stream(self.copy_stream):
                x = x.to(self.device, non_blocking=True)
                self.data_move_event = torch.cuda.Event()
                self.data_move_event.record()
        return x
    
    def register_hooks(self):
        for block_id in range(self.num_blocks):
            next_block_id = (block_id + 1) % self.num_blocks

            def fw_pre_params_copy_hook(module, args):
                with torch.cuda.stream(self.copy_stream):
                    self.blocks[next_block_id] = self.blocks[next_block_id].to(
                        self.device, non_blocking=True)
                    self.param_load_events[next_block_id] = torch.cuda.Event()
                    self.param_load_events[next_block_id].record(self.copy_stream)
                    if self.compute_events[block_id] is not None:
                        self.copy_stream.wait_event(self.compute_events[block_id])
                    self.blocks[block_id] = self.blocks[block_id].to(
                        "cpu", non_blocking=True)
            
            self.blocks[block_id].register_forward_pre_hook(fw_pre_params_copy_hook)


    def forward(self, x: torch.Tensor):
        self.init_load()
        x = self.recv_data(x)
        with torch.cuda.stream(self.compute_stream):        
            if self.data_move_event is not None:
                self.compute_stream.wait_event(self.data_move_event)
            for block_id in range(self.num_blocks):
                self.compute_stream.wait_event(self.param_load_events[block_id])
                x = self.blocks[block_id](x)
                self.compute_events[block_id] = torch.cuda.Event()
                self.compute_events[block_id].record(self.compute_stream)

        return x


class SingleBlockStage(nn.Module):

    def __init__(self, block: nn.Module, device, compute_stream=None, copy_stream=None):
        super().__init__()
        self.block = block
        self.device = device
        self.compute_stream = (compute_stream if compute_stream is not None else torch.cuda.current_stream(device))
        self.copy_stream = (copy_stream if copy_stream is not None else
                            torch.cuda.Stream(device))

        self._init_loaded = False

    def init_load(self):
        if not self._init_loaded:
            with torch.cuda.stream(self.copy_stream):
                self.block = self.block.to(self.device, non_blocking=False)
            self.block_loaded = True

    def forward(self, x: torch.Tensor):
        self.init_load()
        data_move_event = None
        if self.device != x.device:
            with torch.cuda.stream(self.copy_stream):
                x = x.to(self.device)
                data_move_event = torch.cuda.Event()
                data_move_event.record()

        with torch.cuda.stream(self.compute_stream):
            if data_move_event is not None:
                self.compute_stream.wait_event(data_move_event)
            return self.block(x)


class Pipeline:

    def __init__(self, data_iter, stages):
        self.data_iter = data_iter
        self.stages = stages
        self.num_stages = len(self.stages)
        self.model_outputs = Queue()
    
    def create_task(self, input_pack, stage_fn, no_grad):
        def _compute():
            if no_grad:
                with torch.no_grad():
                    return input_pack.call(stage_fn)
            else:
                return input_pack.call(stage_fn)
        
        task = Task(_compute)
        return task
    
    def run(self, limit=None, no_grad=False):
        with spawn_device_workers(self.num_stages) as (in_queues, out_queues):
            flush_count = self.num_stages
            flush = False
            warmup_count = -1
            
            runs = 0
            while True:
                runs += 1
                if not flush:
                    try:
                        batch = next(self.data_iter)
                        inputs, targets = batch
                        if limit is not None:
                            limit -= 1
                            if limit < 0:
                                raise StopIteration()
                        if not isinstance(inputs, List):
                            inputs = [inputs]
                        input_pack = Pack(*inputs)
                        task = self.create_task(input_pack, self.stages[0], no_grad=no_grad)
                        # print(f"Create task stage 0")
                        in_queues[0].put(task)
                        if warmup_count < self.num_stages:
                            warmup_count += 1
                    except StopIteration as e:
                        flush = True
                
                if flush:
                    flush_count -= 1
                    if flush_count < 0:
                        break
                    
                for stage_id in range(self.num_stages):
                    # print(f"stage_id={stage_id}, warmup_count={warmup_count}")
                    if stage_id >= warmup_count:
                        continue
                    if stage_id < self.num_stages - flush_count:
                        continue
                    ok, output_pack = out_queues[stage_id].get()
                    # print(f"stage {stage_id} ok? {ok}")
                    if not ok:
                        trace = ''.join(traceback.format_exception(*output_pack))
                        print("trace:\n", trace)
                        continue
                    if stage_id == self.num_stages - 1:
                        self.model_outputs.put(output_pack.args)
                        print(f"Total outputs: {self.model_outputs.qsize()}")
                        print(f"output: {output_pack.args[0].shape}")
                    else:
                        if output_pack is None:
                            print(f"stage {stage_id+1} output_pack is None")
                        task = self.create_task(output_pack, self.stages[stage_id+1], no_grad=no_grad)
                        # print(f"Create task stage {stage_id+1}")
                        in_queues[stage_id+1].put(task)


class GPTPipeline:
    def __init__(self, embeddings, blocks, lm_head, devices):
        self.embeddings = embeddings
        self.blocks = blocks
        self.lm_head = lm_head
        self.devices = devices
        
        assert len(self.blocks) % (len(self.devices) - 2) == 0
        blocks_per_page = len(self.blocks) // (len(self.devices) - 2)
        assert blocks_per_page > 2
        
        splitted_blocks = [
            blocks[i * blocks_per_page:(i+1) * blocks_per_page]
            for i in range(len(self.devices) - 2)
        ]
        self.num_stages = len(self.devices)
        
        self.copy_streams = {
            device: torch.cuda.Stream(device)
            for device in devices
        }
        self.compute_streams = {
            device: torch.cuda.Stream(device)
            for device in devices
        }
        
        self.pipeline_stages = []
        self.pipeline_stages.append(SingleBlockStage(self.embeddings, self.devices[0], self.compute_streams[self.devices[0]], self.copy_streams[self.devices[0]]))
        for i, blocks in enumerate(splitted_blocks):
            device = self.devices[i+1]
            self.pipeline_stages.append(RotatingBlocksStage(blocks, device, self.compute_streams[device], self.copy_streams[device]))
        self.pipeline_stages.append(SingleBlockStage(self.lm_head, self.devices[-1], self.compute_streams[self.devices[-1]], self.copy_streams[self.devices[-1]]))
        
    def create_pipeline(self, data_iter):
        self.pipeline = Pipeline(data_iter, self.pipeline_stages)
        return self.pipeline
