# Rotary Position Embedding (RoPE) in Triton

This is the implementation example of Rotary Position Embedding (RoPE) kernel using Triton. 

#### RoPE
RoPE is a positional embedding technique that encodes absolute positions using rotation matrices. It naturally includes explicit relative position dependencies in self-attention.

The original paper can be found in [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).


#### Triton
Triton is an open-source programming model developed by OpenAI that empowers developers to efficiently code for GPUs. It facilitates the creation of high-level logic for parallel codes and automates the optimization of data transfer between DRAM and SRAM.

The details about Triton can be found in [Triton official documentation](https://triton-lang.org).

# ⭐ Implementations

## Details

In essence, the primary concept of the implementation is to parallelize across sequence length and head indexes while minimizing data load by reusing the loaded frequency data throughout the batches for each head. Thus, the greater the number of batches, the greater the performance enhancement we can expect. This will be particularly advantageous for data centers where support for multiple batches is essential.

First of all, in the attention in TransformerEngine, the input tensor shapes are as following;
```shell
[seq_len, batch_num, head_num, head_dim] = input.shape
```

By considering the input as the following three-dimensional matrix, we observe that parallelization can be achieved across ```sequence length``` and ```head number```. 



```shell
# Setting grid
grid = (seq_len, head_num)
```

```shell
# Process IDs for 2D
pid_seq = tl.program_id(axis=0)
pid_head = tl.program_id(axis=1)
```

Batches in each head share the same theta as below.

Each head is halved to execute the subsequent calculation. 

To minimize data loading, upon loading frequency data (cos, sin), we use the data to perform calculations with elements across all batches of each head. For a more comprehensive understanding, please consult the figures provided below.

```shell
cos = tl.load(cos_ptr + cos_offset, mask=mask, other=0.0)
sin = tl.load(sin_ptr + sin_offset, mask=mask, other=0.0)

for batch_idx in range(0, batch_num):
    x1 = tl.load(input_ptr + x1_offset, mask=mask, other=0.0)
    x2 = tl.load(input_ptr + x2_offset, mask=mask, other=0.0)

    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos

    tl.store(output_ptr + x1_offset, y1, mask=mask)
    tl.store(output_ptr + x2_offset, y2, mask=mask)
```


## Unit test
The implementation is tested by adjusting the reference test code sourced from NVIDIA's [Transformer Engine](https://github.com/NVIDIA/TransformerEngine/blob/b8eea8aaa94bb566c3a12384eda064bda8ac4fd7/tests/pytorch/test_fused_rope.py). 

To ensure correctness, the following test parameters are utilized.

```shell
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("seq_length", [1024, 2048])
@pytest.mark.parametrize("hidden_size", [64, 128])
@pytest.mark.parametrize("rotary_percent", [1.0])
@pytest.mark.parametrize("margin", [0, 10])
@pytest.mark.parametrize("transpose", [None])
@pytest.mark.parametrize("tensor_format", ["sbhd", "bshd"])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
```


## Performance Benchmarking

The following system was utilized for benchmarking:
  * Ubuntu 22.04, PyTorch 2.2.1 with CUDA 12.1, Python 3.9
  * NVIDIA GPU RTX2070 Super (GDDR6 8GB, 2560 CUDA cores)

The below parameters are used:
  * x_axis: ```[128 * i for i in range(2, 32)]```
  * y_axis: throughput ```GB/s```
  * hidden_size(=head_dim): ```128```
  * batch_size: ```[1, 2, 4, 8]```
  * head_num: ```64```

The CUDA performance converges to ```~250 GB/s``` due to the memory bandwidth bottleneck. Overall, the Triton kernel implementation shows superior performance.

#### Batch 1
For a batch size of 1, Triton initially demonstrates superior performance, but eventually stabilizes around ~260 GB/s, representing only approximately a 4% increase compared to the CUDA result for certain sequence lengths. This marginal improvement occurs because data transfers are necessary for each head of each sequence, resulting in negligible differentiation from GPU performance when handling substantial data loads.

#### Batch 2
Starting from a batch size of 2, the Triton implementation demonstrates superior performance, eventually converging to approximately ~310 GB/s. This represents a notable 24% improvement compared to the CUDA results across all sequence lengths. This observation confirms the anticipated outcome that a higher number of batches corresponds to a more substantial performance enhancement.

#### Batch 4
For a batch size of 4, the performance of the Triton kernel implementation demonstrates approximately a 36% improvement compared to the CUDA results.

#### Batch 8
For a batch size of 8, the Triton kernel implementation demonstrates an enhancement of roughly 44% over the CUDA results.

## Limitation:
Currently, the implementation does not support the ```rotary_percent < 1.0``` test parameter, which applies RoPE only to partial elements, as well as ```transpose=[2,3]```, which alters the underlying memory layout. 



# ⭐ Run
## Prerequisites

To run this RoPE implementation and testing, you need to install specific packages.
Please take note of the environments used and you can create your own based on ```environment.yaml```;

You can also create your own environment;
```bash
conda env create -f environment.yml
```

## Important
Also, please note that you should install transformer-engine separately by using the below
```bash
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

```enviroment.yaml``` does not include the transformer engine package

I used the transformer engine package ```transformer-engine==1.5.0.dev0+a38b291``` which I installed manually from the source due to the freezing issue on my system with buidling wheels.


## Running tests

To run tests, simply execute the following commands:

```shell
# Run Python tests using your local GPU.
$ pytest rope_triton_unittest.py

# Run Triton vs CUDA benchmark using your local GPU.
$ python rope_triton_benchmark.py

```

