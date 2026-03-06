import torch
import sys

def check_gpu_compatibility():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
            
            # Check compatibility
            capability = props.major * 10 + props.minor
            supported = [50, 60, 61, 70, 75, 80, 86, 89, 90]
            
            if capability > max(supported):
                print("  ⚠️ Your GPU is newer than what current PyTorch supports!")
                print("  ✅ Solution: Install PyTorch nightly build")
            elif capability in supported:
                print("  ✅ GPU is fully compatible")
            else:
                print(f"  ❌ GPU is not compatible (supports: {supported})")
    else:
        print("❌ CUDA is not available")
        
        # Check for libraries
        import ctypes
        try:
            ctypes.CDLL("nvcuda.dll")
            print("nvcuda.dll found")
        except:
            print("nvcuda.dll NOT found")

if __name__ == "__main__":
    check_gpu_compatibility()