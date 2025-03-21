import psutil
import time

def bytes_to_gb(bytes):
    return bytes / (1024 ** 3)

def display_memory_usage(memory_blocks):
    program_memory_bytes = len(memory_blocks) * len(memory_blocks[0])
    program_memory_gb = bytes_to_gb(program_memory_bytes)
    os_memory_gb = bytes_to_gb(psutil.virtual_memory().used)
    
    print(f"Program Memory Usage: {program_memory_gb:.2f} GB")
    print(f"OS Memory Usage: {os_memory_gb:.2f} GB")

def consume_memory(max_memory_gb, gb_per_allocation):
    memory_blocks = []
    max_memory_bytes = max_memory_gb * 1024**3
    
    while True:
        try:
            # Allocate specified GB at a time
            for _ in range(gb_per_allocation):
                memory_block = bytearray(1024**3)
                memory_blocks.append(memory_block)
            
            # Print current program memory usage and OS memory usage in GB
            display_memory_usage(memory_blocks)
            
            # Sleep for 2 seconds
            time.sleep(2)
            
            # Check if the maximum memory usage is reached
            if len(memory_blocks) * len(memory_block) >= max_memory_bytes:
                print("Maximum memory usage reached.")
                break
        
        except MemoryError:
            print("Out of memory!")
            break

if __name__ == "__main__":
    max_memory_gb = 96  # Set the maximum memory usage (in GB)
    gb_per_allocation = 4  # Set the number of GBs to allocate at a time
    consume_memory(max_memory_gb, gb_per_allocation)
