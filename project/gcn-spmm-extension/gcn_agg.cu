#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>

#define THREADS 256
// TODO - Modify based on feature size for coalesced access in both cases?

template <typename scalar_t>
__global__ void aggregate_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> feature,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> output,
    const int* __restrict__ src_index,
    const int* __restrict__ tar_index,
    const scalar_t* __restrict__ edge_weight,
    const unsigned int edges_per_block,
    const unsigned int n, const unsigned int num_edge
) {
    if (threadIdx.x < n * edges_per_thread) {
        int feat_id = threadIdx.x % n;
        for (int node_id = blockIdx.x * edges_per_block + threadIdx.x / n; node_id < num_edge; node_id += gridDim.x * edges_per_block) {
            atomicAdd(&output[src_index[node_id]][feat_id], feature[tar_index[node_id]][feat_id] * edge_weight[node_id]);
        }
    }
}

template <typename scalar_t>
__global__ void backward_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> grad_out,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> grad_feature,
    const int* __restrict__ src_index,
    const int* __restrict__ tar_index,
    const scalar_t* __restrict__ edge_weight,
    const unsigned int edges_per_block,
    const unsigned int n, const unsigned int num_edge)
{
    if (threadIdx.x < n * edges_per_thread) {
        int feat_id = threadIdx.x % n;
        for (int node_id = blockIdx.x * edges_per_block + threadIdx.x / n; node_id < num_edge; node_id += gridDim.x * edges_per_block) {
            atomicAdd(&grad_feature[tar_index[node_id]][feat_id], grad_out[src_index[node_id]][feat_id] * edge_weight[node_id]);
        }
    }
}

// feature size is larger than number of threads
template <typename scalar_t>
__global__ void aggregate_large_feature_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> feature,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> output,
    const int* __restrict__ src_index,
    const int* __restrict__ tar_index,
    const scalar_t* __restrict__ edge_weight,
    const unsigned int n, const unsigned int num_edge
) {
    int node_id = blockIdx.x;
    scalar_t weight = edge_weight[node_id];
    for (int feat_id = threadIdx.x; feat_id < n; feat_id += blockDim.x) {
        atomicAdd(&output[src_index[node_id]][feat_id], feature[tar_index[node_id]][feat_id] * weight);
    }
}


template <typename scalar_t>
__global__ void backward_large_feature_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> grad_out,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> grad_feature,
    const int* __restrict__ src_index,
    const int* __restrict__ tar_index,
    const scalar_t* __restrict__ edge_weight,
    const unsigned int n, const unsigned int num_edge)
{
    int node_id = blockIdx.x;
    scalar_t weight = edge_weight[node_id];
    for (int feat_id = threadIdx.x; feat_id < n; feat_id += blockDim.x) {
        atomicAdd(&grad_feature[tar_index[node_id]][feat_id], grad_out[src_index[node_id]][feat_id] * weight);
    }
}

torch::Tensor aggregate_cuda(
    torch::Tensor feature,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    torch::Tensor edge_weight
) {
    unsigned int n = feature.size(1);
    unsigned int num_edge = edge_weight.size(0);

    auto output = torch::empty_like(feature);
    // each thread deals with one feature element
    if (n <= THREADS) {
        // number of edges one thread block can handle
        unsigned int edges_per_block = THREADS / n;
        AT_DISPATCH_FLOATING_TYPES(feature.type(), "aggregate_kernel", ([&] {
            aggregate_kernel<scalar_t> << <(num_edge + edges_per_block - 1) / edges_per_block, THREADS >> > (
                feature.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                src_index.data<int>(), tar_index.data<int>(),
                edge_weight.data<scalar_t>(), edges_per_block, n, num_edge
                );
            }));
        // each thread block handles one edge
    }
    else {
        AT_DISPATCH_FLOATING_TYPES(feature.type(), "aggregate_large_feature_kernel", ([&] {
            aggregate_large_feature_kernel<scalar_t> << <num_edge, THREADS >> > (
                feature.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                src_index.data<int>(), tar_index.data<int>(),
                edge_weight.data<scalar_t>(), n, num_edge
                );
            }));
    }
    return output;
}


std::vector<torch::Tensor> backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor feature,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    torch::Tensor edge_weight)
{
    unsigned int n = grad_out.size(1);
    auto grad_feature = torch::empty_like(grad_out);
    unsigned int num_edge = edge_weight.size(0);
    unsigned int num_node = feature.size(0);
    auto grad_edge_weight = torch::empty_like(edge_weight);

    if (n <= THREADS) {
        unsigned int edges_per_block = THREADS / n;
        AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "backward_kernel", ([&] {
            backward_kernel<scalar_t> << <(num_edge + edges_per_block - 1) / edges_per_block, THREADS >> > (
                grad_out.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_feature.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                src_index.data<int>(), tar_index.data<int>(),
                edge_weight.data<scalar_t>(), edges_per_block, n, num_edge
                );
            }));
    }
    else {
        AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "backward_large_feature_kernel", ([&] {
            backward_large_feature_kernel<scalar_t> << <num_edge, THREADS >> > (
                grad_out.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_feature.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                src_index.data<int>(), tar_index.data<int>(),
                edge_weight.data<scalar_t>(), n, num_edge
                );
            }));
    }
    return { grad_feature, grad_edge_weight };
}