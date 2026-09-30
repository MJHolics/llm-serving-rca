"""같은 GPU를 쓰는 다른 작업을 흉내낸다 — fp16 행렬곱을 끝없이 돌린다(메모리는 수백 MB만)."""
import torch

a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
while True:
    for _ in range(20):
        c = a @ b
    torch.cuda.synchronize()
