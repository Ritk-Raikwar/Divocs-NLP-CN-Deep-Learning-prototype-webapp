import torch
import sys

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(0))
    
    # Test a simple tensor operation on GPU
    print("\nRunning a test computation on GPU...")
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("GPU tensor shape:", z.shape)
    print("GPU test successful!")
else:
    print("CUDA is not available. Using CPU only.")
    
    # Check NVIDIA driver
    import subprocess
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi'], stderr=subprocess.PIPE, universal_newlines=True)
        print("\nNVIDIA-SMI found:")
        print(nvidia_smi)
    except:
        print("\nCould not run nvidia-smi. NVIDIA driver may not be installed properly.")
    
    # Provide troubleshooting info
    print("\nTroubleshooting steps:")
    print("1. Make sure you have an NVIDIA GPU")
    print("2. Ensure NVIDIA drivers are installed")
    print("3. Ensure CUDA Toolkit is installed")
    print("4. Check that you installed PyTorch with CUDA support")
