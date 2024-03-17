import pytest
import torch

from typing import Callable, Dict, Tuple, Union
from transformer_engine.pytorch.attention import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
)

from rope_triton import apply_rotary_pos_emb as rope_triton


def get_tol(dtype: torch.dtype) -> Dict:
    if dtype == torch.bfloat16:
        return dict(atol=1e-2, rtol=1e-2)
    elif dtype == torch.float16:
        return dict(atol=1e-3, rtol=1e-3)
    return dict(atol=1e-5, rtol=1.3e-6)


# Gradient is a broadcasted scalar
def _overlapping_grad(output: torch.Tensor) -> torch.Tensor:
    return output.sum() * 2

# Gradient is a full tensor
def _non_overlapping_grad(output: torch.Tensor) -> torch.Tensor:
    t = torch.ones_like(output)
    return torch.sum(output * t)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("seq_length", [1024, 2048])
@pytest.mark.parametrize("hidden_size", [64, 128])
@pytest.mark.parametrize("rotary_percent", [1.0])
@pytest.mark.parametrize("margin", [0, 10])
@pytest.mark.parametrize("transpose", [None])
@pytest.mark.parametrize("tensor_format", ["sbhd", "bshd"])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
def test_triton_rope(
    dtype: torch.dtype,
    seq_length: int,
    hidden_size: int,
    rotary_percent: float,
    margin: int,
    transpose: Union[Tuple, None],
    tensor_format: str,
    loss_func: Callable,
) -> None:
    device = torch.device("cuda:0")
    batch_size, head_num = 2, 64
    t = torch.rand(
        (seq_length - margin, batch_size, head_num, hidden_size),
        dtype=dtype,
        device=device,
    )
    if tensor_format == "bshd":
        t = t.transpose(0, 1).contiguous()

    if transpose:
        t = t.transpose(*transpose).contiguous().transpose(*transpose)

    t.requires_grad = True

    rotary_pos_emb = RotaryPositionEmbedding(hidden_size, rotary_percent)
    emb = rotary_pos_emb(seq_length)

    # triton
    output_triton = rope_triton(
        t, emb, tensor_format=tensor_format, fused=True
    )

    loss_triton = loss_func(output_triton)
    loss_triton.backward()
    grad_triton = t.grad.detach().clone()
    t.grad = None

    # te
    output_te = apply_rotary_pos_emb(
        t, emb, tensor_format=tensor_format, fused=True,
    )

    loss_te = loss_func(output_te)
    loss_te.backward()
    grad_te = t.grad.detach().clone()
    t.grad = None

    torch.testing.assert_close(output_te, output_triton, **get_tol(dtype))
    torch.testing.assert_close(grad_te, grad_triton, **get_tol(dtype))
    assert output_te.is_contiguous()
