import torch
import triton
import triton.language as tl

from transformer_engine.pytorch.attention import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
)

from rope_triton import apply_rotary_pos_emb as rope_triton


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['SEQ_LEN'],  # argument names to use as an x-axis for the plot
        # different possible values for `x_name`
        x_vals=[128 * i for i in range(2, 32)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        line_vals=[
            'triton',
            'cuda',
        ],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Cuda"
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="rope-performance",
        # values for function arguments not in `x_names` and `y_name`
        args={'HIDDEN_SIZE': 128, 'BATCH_SIZE': 2, 'HEAD_NUM': 64},
    ))
def benchmark(SEQ_LEN, HIDDEN_SIZE, BATCH_SIZE, HEAD_NUM, provider):
    x = torch.rand(
        (SEQ_LEN, BATCH_SIZE, HEAD_NUM, HIDDEN_SIZE),
        dtype=torch.float32,
        device=torch.device("cuda:0"),
    )

    rotary_pos_emb = RotaryPositionEmbedding(HIDDEN_SIZE)
    emb = rotary_pos_emb(SEQ_LEN)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rope_triton(
            x, emb, tensor_format="sbhd", fused=True
        ), quantiles=quantiles)
    if provider == 'cuda':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: apply_rotary_pos_emb(
            x,
            emb,
            tensor_format="sbhd",
            fused=True,
        ), quantiles=quantiles)

    def gbps(ms): return 2 * x.nelement() * \
        x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(show_plots=True, print_data=True)
