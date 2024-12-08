import symmul
import torch

assert torch.cuda.is_available()
device = "cuda"

n = 4096
dtype = torch.bfloat16
A = torch.eye(n, dtype=dtype, device=device)
B = torch.eye(n, dtype=dtype, device=device)
C = torch.zeros((n, n), dtype=dtype, device=device)

symmul.symmul4096_4096(A, B, C)

C_float32 = C.to(torch.float32)
print(C_float32)
