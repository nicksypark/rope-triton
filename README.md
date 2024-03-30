# Rotary Position Embedding (RoPE) optimization in Triton

This is the implementation example of Rotary Position Embedding (RoPE) kernel using Triton. 

#### RoPE
RoPE is a positional embedding technique that encodes absolute positions using rotation matrices. It naturally includes explicit relative position dependencies in self-attention.

The original paper can be found in [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).


#### Triton
Triton is an open-source programming model developed by OpenAI that empowers developers to efficiently code for GPUs. It facilitates the creation of high-level logic for parallel codes and automates the optimization of data transfer between DRAM and SRAM.

The details about Triton can be found in [Triton official documentation](https://triton-lang.org).

# ⭐ Implementation

The kernel implementation is aimed at efficiently executing the computation of RoPE. The computation is demonstrated below from [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).

<p align="center">
<img width="450" alt="image" src="https://github.com/nicksypark/rope-triton/assets/17171917/fe8cbd99-9e40-4cd8-be91-a082831f285d">

</p>
<p align="center">
<img width="450" alt="image" src="https://github.com/nicksypark/rope-triton/assets/17171917/58587da5-ad16-4cbf-8545-850de97f7a13">
</p>

In essence, the primary concept of the implementation is to parallelize across sequence length and head indexes while minimizing data load by reusing the loaded frequency data throughout the batches for each head. Thus, the greater the number of batches, the greater the performance enhancement we can expect. This will be particularly advantageous for data centers where support for multiple batches is essential.

First of all, in the attention in TransformerEngine, the input tensor shapes are as following;
```shell
[seq_len, batch_num, head_num, head_dim] = input.shape
```

### Grid
By considering the input as the following three-dimensional matrix, we observe that parallelization can be achieved across ```sequence length``` and ```head number```. 

<p align="center">
<img width="861" alt="image" src="https://github.com/nicksypark/rope-triton/assets/17171917/75b1d785-2020-4b08-ae01-8360edf1c5f9">


</p>

<p align="center">
<img width="680" alt="image" src="https://github.com/nicksypark/rope-triton/assets/17171917/1e45f118-f2b5-44b6-b522-0cf33b48c39b">
</p>

```shell
# Setting grid
grid = (seq_len, head_num)
```

```shell
# Process IDs for 2D
pid_seq = tl.program_id(axis=0)
pid_head = tl.program_id(axis=1)
```

### Optimization

Each head within batches shares the same theta, as illustrated below. This indicates that theta_1 can be shared across all batches of head 1. (Subsequently, for the computations, each head is divided in half to facilitate the subsequent calculations.)

<p align="center">
<img width="760" alt="image" src="https://github.com/nicksypark/rope-triton/assets/17171917/87104d9a-9f8d-451f-9d9d-9f48d67cca65">
</p>



To minimize data loading, upon loading frequency data (cos, sin), we use the data to perform calculations with elements across all batches of each head. For a more comprehensive understanding, please consult the figures provided below. 

The execution for all heads ranging from 1 to 8 will be performed in parallel.

<p align="center">
<img width="1000" alt="image" src="https://github.com/nicksypark/rope-triton/assets/17171917/88b47723-4466-4d94-a9ee-3b5efab52f13">
</p>




In addition to the heads, the sequence index of 1 to 8 will also be executed in parallel.
<p align="center">
<img width="350" alt="image" src="https://github.com/nicksypark/rope-triton/assets/17171917/b0e5108e-5a22-4666-bba9-8a26b16295b7">
</p>

<p align="center">
<img width="705" alt="image" src="https://github.com/nicksypark/rope-triton/assets/17171917/21c9fdfc-700b-4991-b0dc-04541a9adc55">
</p>

The below code snippet demonstrates the implementation. 
```shell
cos = tl.load(cos_ptr + cos_offset)
sin = tl.load(sin_ptr + sin_offset)

for batch_idx in tl.static_range(0, BATCH_NUM):
    x1 = tl.load(input_ptr + x1_offset)
    x2 = tl.load(input_ptr + x2_offset)

    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos

    tl.store(output_ptr + x1_offset, y1)
    tl.store(output_ptr + x2_offset, y2)
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
  * Ubuntu 22.04
  * NVIDIA GPU RTX2070 Super (GDDR6 8GB, 4 MB L2 cache, 2560 CUDA cores)
  * transformer-engine==1.5.0.dev
  * PyTorch 2.2.1 with CUDA 12.1, Python 3.9.18

The below parameters are used:
  * x_axis: ```[128 * i for i in range(2, 32)]```
  * y_axis: throughput ```GB/s```
  * hidden_size (=head_dim): ```128```
  * batch_size: ```[1, 2, 4, 8]```
  * head_num: ```64```



Note: The below is the illustration of the Transformer Engine implementation

<p align="center">
<img width="748" alt="image" src="https://github.com/nicksypark/rope-triton/assets/17171917/68718c3c-08fd-481c-81ad-3da8bdbbb510">
</p>

The CUDA performance converges to ```~250 GB/s``` due to the memory bandwidth bottleneck. Overall, the Triton kernel implementation shows superior performance.

#### Batch 1
For a batch size of 1, Triton initially demonstrates superior performance, but eventually stabilizes around ~260 GB/s, representing only approximately a 4% increase compared to the CUDA result for certain sequence lengths. This marginal improvement occurs because data transfers are necessary for each head of each sequence, resulting in negligible differentiation from GPU performance when handling substantial data loads.

![image](https://github.com/nicksypark/rope-triton/assets/17171917/810b8dce-e226-4352-8b12-d8afba026e54)

#### Batch 2
Starting from a batch size of 2, the Triton implementation demonstrates superior performance, eventually converging to approximately ~310 GB/s. This represents a notable 24% improvement compared to the CUDA results across all sequence lengths. This observation confirms the anticipated outcome that a higher number of batches corresponds to a more substantial performance enhancement.

![image](https://github.com/nicksypark/rope-triton/assets/17171917/e17451d8-0aab-45f8-85c9-ca3dedea38ec)

#### Batch 4
For a batch size of 4, the performance of the Triton kernel implementation demonstrates approximately a 36% improvement compared to the CUDA results.

![image](https://github.com/nicksypark/rope-triton/assets/17171917/a494c7cd-2eaf-4124-a5ef-beb82709dc3d)

#### Batch 8
For a batch size of 8, the Triton kernel implementation demonstrates an enhancement of roughly 44% over the CUDA results.

![image](https://github.com/nicksypark/rope-triton/assets/17171917/93d9a4ac-efc3-4372-bb95-abc38b380ced)

## Limitation:
Currently, the implementation does not support the ```rotary_percent < 1.0``` test parameter, which applies RoPE only to partial elements, as well as ```transpose=[2,3]```, which alters the underlying memory layout. ```sbhd``` and ```bshd``` input tensor formats are only supported.



# ⭐ Run
## Prerequisites

On Ubuntu, to run this RoPE implementation and testing, you need to install specific packages.
Please take note of the environments used and you can create your own based on ```environment.yml```;

```bash
conda env create -f environment.yml -p /home/<user>/anaconda3/envs/<envname>
```

## Important
Also, please note that you should install ```transformer-engine``` separately by using the below;
```bash
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

The included ```enviroment.yml``` does not include the transformer engine package.
I used the transformer engine package ```transformer-engine==1.5.0.dev0+a38b291``` which I installed manually from the source due to the freezing issue on my system with buidling wheels.


## Running tests

To run tests, simply execute the following commands:

```shell
# Run Python tests using your local GPU.
$ pytest rope_triton_unittest.py

# Run Triton vs CUDA benchmark using your local GPU.
$ python rope_triton_benchmark.py

```

For performance benchmarking, you can also adjust the parameters provided below.
```shell
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

```



