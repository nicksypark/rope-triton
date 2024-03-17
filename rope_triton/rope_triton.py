import torch
import triton
import triton.language as tl
from typing import Tuple, Union


@triton.jit
def rope_kernel_fw(input_ptr, in_seq_len_stride, in_batch_stride,
                   output_ptr, cos_ptr, sin_ptr, cos_stride, sin_stride,
                   seq_len, batch_num, head_dim,
                   BLOCK_SIZE: tl.constexpr):
    pid_seq = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    head_dim_offset = tl.arange(0, BLOCK_SIZE)  # [0:head_dim/2]
    head_dim_mid = head_dim // 2
    mask = head_dim_offset < head_dim_mid

    cos_offset = (pid_seq % seq_len) * cos_stride + head_dim_offset
    sin_offset = (pid_seq % seq_len) * sin_stride + head_dim_offset

    cos = tl.load(cos_ptr + cos_offset, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + sin_offset, mask=mask, other=0.0)

    for batch_idx in range(0, batch_num):
        x1_offset = pid_seq * in_seq_len_stride + batch_idx * \
            in_batch_stride + pid_head * head_dim + head_dim_offset
        x2_offset = pid_seq * in_seq_len_stride + batch_idx * in_batch_stride + \
            pid_head * head_dim + head_dim_mid + head_dim_offset

        x1 = tl.load(input_ptr + x1_offset, mask=mask, other=0.0)
        x2 = tl.load(input_ptr + x2_offset, mask=mask, other=0.0)

        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        tl.store(output_ptr + x1_offset, y1, mask=mask)
        tl.store(output_ptr + x2_offset, y2, mask=mask)
    return


@triton.jit
def rope_kernel_bw(input_ptr, in_seq_len_stride, in_batch_stride,
                   output_ptr, cos_ptr, sin_ptr, cos_stride, sin_stride,
                   seq_len, batch_num, head_dim,
                   BLOCK_SIZE: tl.constexpr):
    pid_seq = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    head_dim_offset = tl.arange(0, BLOCK_SIZE)  # [0:head_dim/2]
    head_dim_mid = head_dim // 2
    mask = head_dim_offset < head_dim_mid

    cos_offset = (pid_seq % seq_len) * cos_stride + head_dim_offset
    sin_offset = (pid_seq % seq_len) * sin_stride + head_dim_offset

    cos = tl.load(cos_ptr + cos_offset, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + sin_offset, mask=mask, other=0.0)

    for batch_idx in range(0, batch_num):
        x1_offset = pid_seq * in_seq_len_stride + batch_idx * \
            in_batch_stride + pid_head * head_dim + head_dim_offset
        x2_offset = pid_seq * in_seq_len_stride + batch_idx * in_batch_stride + \
            pid_head * head_dim + head_dim_mid + head_dim_offset

        x1 = tl.load(input_ptr + x1_offset, mask=mask, other=0.0)
        x2 = tl.load(input_ptr + x2_offset, mask=mask, other=0.0)

        y1 = x1 * cos - x2 * -sin
        y2 = x1 * -sin + x2 * cos

        tl.store(output_ptr + x1_offset, y1, mask=mask)
        tl.store(output_ptr + x2_offset, y2, mask=mask)

    return


class FusedRoPEFucnTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        tensor_format: str = "sbhd",
        cu_seqlens: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if tensor_format == "bshd":
            t = t.transpose(0, 1)
        elif tensor_format != "sbhd":
            raise ValueError(f"Unsupported tensor_format: {tensor_format}.")

        seq_len, batch_num, head_num, head_dim = t.shape
        output = torch.empty_like(t)

        BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)

        grid = (seq_len, head_num)

        freqs = freqs[:seq_len]
        cos = torch.cos(freqs).to(t.dtype)
        sin = torch.sin(freqs).to(t.dtype)

        rope_kernel_fw[grid](t,
                             t.stride(0),
                             t.stride(1),
                             output,
                             cos,
                             sin,
                             cos.stride(0),
                             sin.stride(0),
                             seq_len,
                             batch_num,
                             head_dim,
                             BLOCK_SIZE)

        ctx.cos = cos
        ctx.sin = sin
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.tensor_format = tensor_format

        if tensor_format == "bshd":
            return output.transpose(0, 1)
        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        if ctx.tensor_format == "bshd":
            grad_output = grad_output.transpose(0, 1)
        elif ctx.tensor_format != "sbhd":
            raise ValueError(
                f"Unsupported tensor_format: {ctx.tensor_format}.")

        seq_len, batch_num, head_num, head_dim = grad_output.shape
        grad_input = torch.empty_like(grad_output)

        grid = (seq_len, head_num)

        rope_kernel_bw[grid](grad_output.clone(),
                             grad_input.stride(0),
                             grad_input.stride(1),
                             grad_input,
                             ctx.cos,
                             ctx.sin,
                             ctx.cos.stride(0),
                             ctx.sin.stride(0),
                             seq_len,
                             batch_num,
                             head_dim,
                             ctx.BLOCK_SIZE)

        if ctx.tensor_format == "bshd":
            return grad_input.transpose(0, 1), None, None, None, None

        return grad_input, None, None, None, None


def apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    fused: bool = False,
    cu_seqlens: Union[torch.Tensor, None] = None,
) -> torch.Tensor:
    if fused:
        return FusedRoPEFucnTriton.apply(t, freqs, tensor_format, cu_seqlens)
    else:
        "Only fused option is supported"
