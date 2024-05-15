# gcn-cuda-kernel

Graph Convolutional Networks learn representations that account for the graphical structure of the data
[GCN](https://snap.stanford.edu/class/cs224w-2020/slides/06-GNN1.pdf) involves 3 stages:
1. Combination
2. Aggregation
3. Graph Processing

This project includes a custom CUDA implementation for the Sparse Matrix Multiplication operation in the aggregation stage.

This custom CUDA kernel is bound to standard PyTorch code with PyBind and tested on the [Planetoid dataset] (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid) on the task of classifying a publication based a graphical structure built by linked citations.
