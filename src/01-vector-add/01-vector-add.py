import torch

import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)  # cuda BlockIdx.x
    block_start = pid * BLOCK_SIZE  # data start
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)  # pre alloc output tensor
    assert x.is_cuda and y.is_cuda and output.is_cuda

    n_elements = output.numel()  # element number

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # kernel function
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")

    # pytorch version
    output_torch = x + y
    print(output_torch)

    output_triton = add(x, y)
    print(output_triton)

    max_diff = torch.max(torch.abs(output_torch - output_triton))
    print(f"The max difference between torch and triton is {max_diff}")
