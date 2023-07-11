
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <vector>

namespace py = pybind11;


extern torch::Tensor aggregate_cuda(
    torch::Tensor feature,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    torch::Tensor edge_weight
);


torch::Tensor aggregate(
    torch::Tensor feature,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    torch::Tensor edge_weight)
{
    return aggregate_cuda(feature, src_index, tar_index, edge_weight);
}


std::vector<torch::Tensor> backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor feature,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    torch::Tensor edge_weight
);


std::vector<torch::Tensor> backward(
    torch::Tensor grad_out,
    torch::Tensor feature,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    torch::Tensor edge_weight
)
{
    return backward_cuda(grad_out, feature, src_index, tar_index, edge_weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gcn_agg_cuda", &aggregate, "GCN aggregation with sparse matrix multiply forward");
    m.def("gcn_agg_cuda", &backward, "GCN aggregation with sparse matrix multiply backward");
}
