import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "N/A")
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# Additional info that might be helpful
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device capability:", torch.cuda.get_device_capability(0))
    
    # Test a small tensor operation on GPU
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    z = x @ y  # Matrix multiplication
    end.record()
    
    # Waits for everything to finish running
    torch.cuda.synchronize()
    
    print(f"Matrix multiplication time: {start.elapsed_time(end):.2f} ms")
    print("GPU test successful!")
