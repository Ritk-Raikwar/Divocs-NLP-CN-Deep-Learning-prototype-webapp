import subprocess
import time
import os
import platform
import argparse

def get_gpu_info_windows():
    """Get GPU info using nvidia-smi on Windows"""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
             '--format=csv,noheader,nounits'],
            universal_newlines=True
        )
        values = result.strip().split(',')
        if len(values) >= 4:
            gpu_util = float(values[0].strip())
            mem_used = float(values[1].strip())
            mem_total = float(values[2].strip())
            temp = float(values[3].strip())
            return f"GPU: {gpu_util:.1f}% | Memory: {mem_used:.0f}/{mem_total:.0f} MB | Temp: {temp:.1f}°C"
        return "Error parsing GPU info"
    except Exception as e:
        return f"Error getting GPU info: {e}"

def get_gpu_info_linux():
    """Get GPU info using nvidia-smi on Linux"""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
             '--format=csv,noheader,nounits'],
            universal_newlines=True
        )
        values = result.strip().split(',')
        if len(values) >= 4:
            gpu_util = float(values[0].strip())
            mem_used = float(values[1].strip())
            mem_total = float(values[2].strip())
            temp = float(values[3].strip())
            return f"GPU: {gpu_util:.1f}% | Memory: {mem_used:.0f}/{mem_total:.0f} MB | Temp: {temp:.1f}°C"
        return "Error parsing GPU info"
    except Exception as e:
        return f"Error getting GPU info: {e}"

def clear_screen():
    """Clear the console screen based on OS"""
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')

def main():
    parser = argparse.ArgumentParser(description='Monitor GPU usage')
    parser.add_argument('-i', '--interval', type=float, default=2.0,
                      help='Update interval in seconds (default: 2.0)')
    args = parser.parse_args()
    
    print("Starting GPU monitoring (Press Ctrl+C to stop)...")
    
    try:
        while True:
            clear_screen()
            print(f"=== GPU MONITORING (Updated every {args.interval} seconds) ===")
            if platform.system() == "Windows":
                print(get_gpu_info_windows())
            else:
                print(get_gpu_info_linux())
            print("Press Ctrl+C to stop monitoring")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()
