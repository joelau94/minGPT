import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Optional

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

    def __init__(self, blocks: List[nn.Module], device, copy_stream=None):
        super().__init__()
        self.blocks = blocks
        self.device = device

        self.num_blocks = len(self.blocks)
        assert self.num_blocks > 2

        self.compute_stream = torch.cuda.Stream(device)
        self.copy_stream = (copy_stream if copy_stream is not None else
                            torch.cuda.Stream(device))

        self._init_loaded = False

    def init_load(self):
        if not self._init_loaded:
            with torch.cuda.stream(self.copy_stream):
                self.blocks[0] = self.blocks[0].to(self.device,
                                                   non_blocking=False)
            self._init_loaded = True

    def forward(self, x: torch.Tensor):
        self.init_load()
        for block_id in range(self.num_blocks):
            next_block_id = (block_id + 1) % self.num_blocks

            data_move_event = None
            if self.device != x.device:
                with torch.cuda.stream(self.copy_stream):
                    x = x.to(self.device)
                    data_move_event = torch.cuda.Event()
                    data_move_event.record()

            with torch.cuda.stream(self.compute_stream):
                if data_move_event is not None:
                    self.compute_stream.wait_event(data_move_event)
                x = self.blocks[block_id](x)
                compute_event = torch.cuda.Event()
                compute_event.record(self.compute_stream)

            with torch.cuda.stream(self.copy_stream):
                self.blocks[next_block_id] = self.blocks[next_block_id].to(
                    self.device, non_blocking=True)
                self.copy_stream.wait_event(compute_event)
                self.blocks[block_id] = self.blocks[block_id].to(
                    "cpu", non_blocking=True)

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

    def forward(self, *args, **kwargs):
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
            return self.block(*args, **kwargs)


class Pipeline:

    def __init__(self, batches, stages, devices, copy_streams: Optional[List[torch.cuda.Stream]] = None):
        self.batches = batches
        self.stages = stages
        self.devices = devices
        self.copy_streams = copy_streams
        
    def schedules
