"""
Output model complexity
"""
import time

import torch
from torchstat import stat
from torchsummary import summary

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd

device = "cpu"  # default cpu
width = 320
height = 240

# fd = create_mb_tiny_fd(2)
fd = create_Mb_Tiny_RFB_fd(2)

print(fd)
fd.eval()
fd.to(device)
x = torch.randn(1, 3, width, height).to(device)

summary(fd.to("cuda"), (3, width, height))

from ptflops import get_model_complexity_info

flops, params = get_model_complexity_info(fd.to(device), (3, width, height), print_per_layer_stat=True, as_strings=True)
print("FLOPS:", flops)
print("PARAMS:", params)

for i in range(5):
    time_time = time.time()
    features = fd(x)
    print("inference time :{} s".format(time.time() - time_time))

stat(fd, (3, width, height))
