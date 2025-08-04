import os
import sys
import subprocess
import platform

def run_command(command):
    """Run a command and return output"""
    try:
        result = subprocess.check_output(
            command, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            shell=True
        )
        return result.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.output.strip()}"
    except Exception as e:
        return f"Error: {str(e)}"

def check_nvidia_driver():
    """Check if NVIDIA driver is installed"""
    print("Checking NVIDIA driver...")
    result = run_command("nvidia-smi")
    if "NVIDIA-SMI" in result:
        print("✓ NVIDIA driver found")
        print(result)
        return True
    else:
        print("✗ NVIDIA driver not found or not working properly")
        print(result)
        return False

def check_cuda_installation():
    """Check if CUDA is installed"""
    print("\nChecking CUDA installation...")
    result = run_command("nvcc --version")
    if "Cuda compilation tools" in result:
        print("✓ CUDA found")
        print(result)
        return True
    else:
        print("✗ CUDA not found in PATH")
        print(result)
        
        # Check common CUDA installation paths
        cuda_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin\nvcc.exe",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\nvcc.exe",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\nvcc.exe",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\nvcc.exe"
        ]
        
        for path in cuda_paths:
            if os.path.exists(path):
                print(f"Found CUDA at: {path}")
                print(f"But it's not in your PATH. Consider adding the directory to your PATH.")
                return False
        
        print("Could not find CUDA installation in common locations.")
        return False

def check_pytorch_cuda():
    """Check if PyTorch can access CUDA"""
    print("\nChecking PyTorch CUDA support...")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print("✓ PyTorch CUDA is available")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("✗ PyTorch CUDA is not available")
            if "+" in torch.__version__:
                if "+cpu" in torch.__version__:
                    print("You have the CPU-only version of PyTorch installed.")
                    print("Reinstall PyTorch with CUDA support.")
                else:
                    cuda_version = torch.__version__.split("+")[1]
                    print(f"You have PyTorch with {cuda_version} support, but it can't access your GPU.")
            else:
                print("You might have PyTorch installed without explicit CUDA support.")
            return False
    except ImportError:
        print("✗ PyTorch is not installed")
        return False

def check_path():
    """Check if CUDA directories are in PATH"""
    print("\nChecking PATH for CUDA directories...")
    path_parts = os.environ.get('PATH', '').split(os.pathsep)
    cuda_paths = [p for p in path_parts if 'cuda' in p.lower() or 'nvidia' in p.lower()]
    
    if cuda_paths:
        print("✓ Found CUDA/NVIDIA directories in PATH:")
        for p in cuda_paths:
            print(f"  - {p}")
    else:
        print("✗ No CUDA directories found in PATH")
        print("You should add the CUDA bin directory to your PATH")

def print_summary(driver_ok, cuda_ok, pytorch_ok):
    """Print summary of findings"""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if driver_ok and cuda_ok and pytorch_ok:
        print("✓ Everything looks good! Your system is properly set up for PyTorch with CUDA.")
    else:
        print("There are issues with your setup:")
        
        if not driver_ok:
            print("✗ NVIDIA driver issue - Install or update your NVIDIA drivers")
        
        if not cuda_ok:
            print("✗ CUDA issue - Install CUDA Toolkit or add it to PATH")
        
        if not pytorch_ok:
            print("✗ PyTorch CUDA issue - Reinstall PyTorch with CUDA support")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    print("\nRecommended steps:")
    if not driver_ok:
        print("1. Download and install latest NVIDIA drivers from https://www.nvidia.com/Download/index.aspx")
    
    if not cuda_ok:
        print("1. Download and install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads")
        print("2. Add CUDA bin directory to your PATH:")
        print('   $env:Path += ";C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin"')
    
    if not pytorch_ok:
        print("1. Reinstall PyTorch with CUDA support:")
        print("   pip uninstall -y torch torchvision")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    print(f"System: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"Python: {sys.version.split()[0]}")
    
    driver_ok = check_nvidia_driver()
    cuda_ok = check_cuda_installation()
    check_path()
    pytorch_ok = check_pytorch_cuda()
    
    print_summary(driver_ok, cuda_ok, pytorch_ok)
