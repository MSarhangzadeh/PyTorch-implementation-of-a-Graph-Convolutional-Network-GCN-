import torch 
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    
    """
    Graph Convolutional Network layer as described in Kipf & Welling (2017).

    Args:
        in_channels (int): Number of input node features.
        out_channels (int): Number of output node features.

    Forward Inputs:
        x (Tensor): Node feature matrix with shape [num_nodes, in_channels].
        edge_index (LongTensor): Graph connectivity in COO format with shape [2, num_edges].

    Returns:
        Tensor: Updated node features with shape [num_nodes, out_channels].
    """
    
    def __init__(self, in_channels, out_channels): 
        super(GCNConv, self).__init__(aggr='add')
        self.lin = Linear(in_channels, out_channels)
        self.bias = Parameter(torch.empty(out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()
    
    def forward(self, x, edge_index):
        
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col = edge_index
        deg = degree(col,x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        out = self.propagate(edge_index, x=x, norm=norm)
        out = out + self.bias
        
        return out 

    def message(self, x_j, norm):
        
        return norm.view(-1,1)* x_j
    