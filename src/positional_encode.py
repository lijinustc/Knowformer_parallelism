import torch_geometric as pyg
from torch_geometric.utils import add_self_loops
import torch 

def PE(edge_index):
    edge_index=edge_index[:,[0,2]].transpose(0,1).to('cpu')
    data=pyg.data.Data(torch.zeros((14541,32)),edge_index=add_self_loops(edge_index)[0])
    pencoder=pyg.transforms.AddRandomWalkPE(32,'pe')
    return pencoder(data).pe