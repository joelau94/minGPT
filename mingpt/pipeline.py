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

    def forward(self, x: torch.Tensor):
        for block_id in range(self.num_blocks):
            local_block_id = block_id % self.num_devices
            device_id = block_id // self.num_devices
            device = self.devices[device_id]

            next_block_id = (device_id * self.blocks_per_stage +
                             (local_block_id + 1) % self.blocks_per_stage)

            params_copy_stream = self.params_copy_streams[device]
            compute_stream = self.compute_streams[device]
            
            if device != x.device:
                with torch.cuda.stream(self.data_copy_streams[device]):
                    x = x.to(device)

            with torch.cuda.stream(compute_stream):
                x = self.blocks[block_id](x)
                compute_done_event = torch.cuda.Event()

            with torch.cuda.stream(params_copy_stream):
                self.blocks[next_block_id] = self.blocks[next_block_id].to(device, non_blocking=True)
                compute_done_event.record(params_copy_stream)
                self.blocks[block_id] = self.blocks[block_id].to("cpu", non_blocking=True)
                
        return x


def create_gpu_devices(gpu_ids):
    return [torch.device(f"cuda:{i}") for i in gpu_ids]
