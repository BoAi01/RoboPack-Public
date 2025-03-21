import torch
import time

# Function to benchmark tensor transfer
def benchmark_tensor_transfer(tensor, source_device, target_device):
    start_time = time.time()
    tensor = tensor.to(target_device)
    end_time = time.time()
    transfer_time = end_time - start_time
    return transfer_time

# Set the number of repetitions
N = 100

# Define tensor size and data type
tensor_size = (1000, 100)
dtype = torch.float32

# Create a tensor on the source device (e.g., GPU)
source_device = torch.device("cpu")
tensor_on_source = torch.randn(tensor_size, dtype=dtype, device=source_device)

# Define the target device (e.g., CPU)
target_device = torch.device("cuda:0")

# Repeat N times and measure transfer time
total_transfer_time = 0
for i in range(N):
    transfer_time = benchmark_tensor_transfer(tensor_on_source, source_device, target_device)
    total_transfer_time += transfer_time

average_transfer_time = total_transfer_time / N

print(f"Average transfer time for {N} repetitions: {average_transfer_time:.6f} seconds for tensor shaped: {tensor_size}")
