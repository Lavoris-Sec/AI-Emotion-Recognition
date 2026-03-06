import torch
import sys

print("="*50)
print("RTX COMPATIBILITY CHECK")
print("="*50)

print(f"PyTorch version: {torch.__version__}")
print(f"Python version: {sys.version}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Check compute capability
    props = torch.cuda.get_device_properties(0)
    print(f"Compute Capability: {props.major}.{props.minor}")
    
    # Test simple GPU operation
    try:
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = torch.mm(x, y)
        print("\n✅ TEST PASSED: Matrix multiplication on GPU works!")
        print(f"Result on device: {z.device}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
else:
    print("❌ CUDA NOT available!")