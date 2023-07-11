
import os
import torch

#pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
#pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
#pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch_geometric
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
import numpy
#https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/message_passing.html#MessagePassing
#https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gcn_conv.html#GCNConv
#https://github.com/apuaaChen/gcnLib/blob/c168423bf76d38f1197cbc85fd71b6a3962fcd0c/fuseGNN/convs/gcn_conv.py
import torch
import torch_scatter
import torch.nn.functional as F
#from scipy.sparse import coo_matrix
import torch
import torch_scatter
import torch.nn.functional as F
import gcn_agg_cuda

os.environ['TORCH'] = torch.__version__
data = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
criterion = torch.nn.CrossEntropyLoss()


# reference implementation - PyGeometric
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(data.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, data.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

"""**Custom GCN model - No GPU kernel**"""
class NaiveGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(NaiveGCNConv, self).__init__()
        self.tid, self.sid = (1, 0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edges = None
        self.deg = None
        self.weight = None
        self.num_edges = None
        self.dense = torch.nn.Linear(in_features=in_channels, out_features=out_channels, bias=bias)
    
    def forward(self, x, edge_index):
        src_index, tar_index = (edge_index[self.sid], edge_index[self.tid])
        # combination
        x = self.dense(x)
        # graph processing
        self.get_adj(src_index, tar_index, x.size(0))
        # aggregation
        return self.propagate(x, src_index, tar_index, edge_index)
    
    def propagate(self, feature, src_index, tar_index, edge_index):
        adj = torch.sparse_coo_tensor(edge_index, self.weight)
        out = torch.mm(adj, feature)
        out += feature * self.deg.pow(-1).unsqueeze(1)
        return out
        
    def get_adj(self, src_index, tar_index, num_nodes):
        self.num_edges = src_index.size(0)
        self.processing_edge(src_index, tar_index, num_nodes)
    
    def processing_edge(self, src_index, tar_index, num_nodes):
        edge_weight = torch.ones(size=(src_index.size(0),), dtype=torch.float32, device=src_index.device)
        deg = torch_scatter.scatter_add(src=edge_weight, index=src_index, dim=0, dim_size=num_nodes) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[src_index] * edge_weight * deg_inv_sqrt[tar_index]
        self.deg = deg
        self.weight = edge_weight
        self.num_edges = src_index.size(0)

class NaiveGCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = NaiveGCNConv(data.num_features, hidden_channels)
        self.conv2 = NaiveGCNConv(hidden_channels, data.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


"""**Custom GCN model - Custom GPU kernel for sparse matrix multiplication**"""

class GCNAggregate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feature, src_index, tar_index, edge_weight):
        out = gcn_agg_cuda.aggregate(feature, src_index, tar_index, edge_weight)
        ctx.save_for_backward(src_index, tar_index, edge_weight)
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        src_index, tar_index, edge_weight = ctx.saved_tensors
        grad_features, _, _2 = gcn_agg_cuda.backward(grad_out, *ctx.saved_tensors)
        return grad_features, None, None, None, None

class CustomGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(CustomGCNConv, self).__init__()
        self.tid, self.sid = (1, 0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edges = None
        self.deg = None
        self.weight = None
        self.num_edges = None
        # The combination stage after aggregation
        self.dense = torch.nn.Linear(in_features=in_channels, out_features=out_channels, bias=bias)
    
    def forward(self, x, edge_index):
        src_index, tar_index = (edge_index[self.sid], edge_index[self.tid])
        x = self.dense(x)
        self.get_adj(src_index, tar_index, x.size(0))
        return self.propagate(x, src_index, tar_index, edge_index)
        
    def get_adj(self, src_index, tar_index, num_nodes):
        self.num_edges = src_index.size(0)
        self.processing_edge(src_index, tar_index, num_nodes)
    
    def processing_edge(self, src_index, tar_index, num_nodes):
        edge_weight = torch.ones(size=(src_index.size(0),), dtype=torch.float32, device=src_index.device)
        deg = torch_scatter.scatter_add(src=edge_weight, index=src_index, dim=0, dim_size=num_nodes) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[src_index] * edge_weight * deg_inv_sqrt[tar_index]
        self.deg = deg
        self.weight = edge_weight
        self.num_edges = src_index.size(0)
        
    def propagate(self, feature, src_index, tar_index, edge_index):
       # Call CUDA kernel wrapper
        out = GCNAggregate.apply(feature, src_index, tar_index, self.weight)
        out += feature * self.deg.pow(-1).unsqueeze(1)
        return out

class CustomGCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = CustomGCNConv(data.num_features, hidden_channels)
        self.conv2 = CustomGCNConv(hidden_channels, data.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x, data.edge_index)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc




if __name__ == "__main__":
    ref_model = GCN(hidden_channels=16)
    naive_model = NaiveGCN(hidden_channels=16)
    custom_model = CustomGCN(hidden_channels=16)
    models = [ref_model, naive_model, custom_model]
    tests = []
    for model in models:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        for epoch in range(1, 101):
            loss = train(optimizer)
        test_acc = test()
        print(test_acc)
        tests.append(test_acc)
    # check for equivalence
    if len(set(tests)) == 1:
        print('Pass')
            
            
