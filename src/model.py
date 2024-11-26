import math
from copy import deepcopy

import numpy as np
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rspmm import generalized_rspmm
from concurrent.futures import ThreadPoolExecutor


class KnowformerFFN(nn.Module):
    def __init__(self, hidden_dim, drop):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drop = drop
        
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.fc2(x)
        return x


class KnowformerQKLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.mlp_out = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), 
                                     nn.Linear(self.hidden_dim, self.hidden_dim))
        self.alpha = nn.Parameter(torch.empty(1, self.hidden_dim))
        nn.init.normal_(self.alpha)
        self.eps = torch.nn.Parameter(torch.tensor([0.0]))
        self.norm = nn.LayerNorm(self.hidden_dim)
        
    def forward(self, x, z, graph, graph_mask):
        batch_size = x.size(0)
        V = x.size(1)
        R = z.size(1)
        
        # define some functions
        split = lambda t: einops.rearrange(t, 'b l d -> l (b d)')
        merge = lambda t: einops.rearrange(t, 'l (b d) -> b l d', b=batch_size)
        
        edge_index = graph.edge_index if graph_mask is None else graph.edge_index[graph_mask]
        
        # the rspmm cuda kernel from torchdrug 
        # https://torchdrug.ai/docs/api/layers.html#torchdrug.layers.functional.generalized_rspmm
        # reduce memory complexity from O(|E|d) to O(|V|d)
        output = generalized_rspmm(edge_index[:, [0, 2]].transpose(0, 1), edge_index[:, 1], torch.ones_like(edge_index[:, 0]).float(),
                                   relation=split(z.float()), input=split(x.float()))
        output = merge(output)
        
        x_shortcut = x
        x = self.mlp_out(output + self.alpha * x)
        x = self.norm(x)
        x = x + x_shortcut
        return x
    
    
class KnowformerVLayer(nn.Module):
    def __init__(self, num_relation, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relation = num_relation
        
        self.fc_pna = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_z = nn.Linear(self.hidden_dim, self.hidden_dim*self.num_relation)
        self.fc_out = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), 
                                    nn.Linear(self.hidden_dim, self.hidden_dim))
        self.beta = nn.Parameter(torch.empty(1, self.hidden_dim))
        nn.init.normal_(self.beta)
        self.eps = torch.nn.Parameter(torch.tensor([0.0]))
        self.norm = nn.LayerNorm(self.hidden_dim)
        
        self.fc_readout_i = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_readout_o = nn.Linear(self.hidden_dim, self.hidden_dim)
        
    def forward(self, x, z, r_index, graph, graph_mask=None):
        batch_size = x.size(0)
        V = x.size(1)
        R = self.num_relation
        
        # define some functions
        split = lambda t: einops.rearrange(t, 'b l d -> l (b d)')
        merge = lambda t: einops.rearrange(t, 'l (b d) -> b l d', b=batch_size)
        
        z = einops.rearrange(self.fc_z(z), 'b (r d) -> b r d', r=R)
        
        edge_index = graph.edge_index if graph_mask is None else graph.edge_index[graph_mask]
        
        # the rspmm cuda kernel from torchdrug 
        # https://torchdrug.ai/docs/api/layers.html#torchdrug.layers.functional.generalized_rspmm
        # reduce memory complexity from O(|E|d) to O(|V|d)
        output = generalized_rspmm(edge_index[:, [0, 2]].transpose(0, 1), edge_index[:, 1], torch.ones_like(edge_index[:, 0]).float(),
                                   relation=split(z.float()), input=split(x.float()))
        output = merge(output)

        x_shortcut = x
        x = self.fc_out(output + self.beta * x) 
        x = self.norm(x)
        x = x + x_shortcut
        return x
    

class KnowformerLayer(nn.Module):
    def __init__(self,layer_index, num_relation, num_qk_layer, num_v_layer, hidden_dim, num_heads, drop):
        super().__init__()
        self.num_relation = num_relation
        self.num_qk_layer = num_qk_layer
        self.num_v_layer = num_v_layer
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.drop = drop
        self.head_dim=hidden_dim // num_heads
        # define for getting proper device
        self.dummy_param = nn.Parameter(torch.zeros(1))
        
        layer = KnowformerVLayer(self.num_relation, self.hidden_dim)
        self.v_layers = nn.ModuleList([deepcopy(layer) for _ in range(self.num_v_layer)])
        #layer = KnowformerQKLayer(self.hidden_dim)
        #self.qk_layers = nn.ModuleList([deepcopy(layer) for _ in range(self.num_qk_layer)])
        
        #self.fc_qk_x = nn.Sequential(nn.Linear(self.hidden_dim+1, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.fc_v_x = nn.Sequential(nn.Linear(self.hidden_dim*2, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.fc_attn = nn.Linear(self.hidden_dim//self.num_heads*2, self.hidden_dim//self.num_heads)
        self.fc_attn_value = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        #self.fc_to_qk = nn.Linear(self.hidden_dim, self.hidden_dim*2*2)
        self.fc_to_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        #self.fc_qk_z = nn.Linear(self.hidden_dim, self.hidden_dim*self.num_relation)
        
        self.ffn = KnowformerFFN(self.hidden_dim, self.drop)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.attn_norm = nn.LayerNorm(self.hidden_dim)
        
        self.lambda_init=0.8 - 0.6 * math.exp(-0.3 * layer_index)
        
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        
        self.W_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_k = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_v = nn.Linear(self.hidden_dim, self.hidden_dim)
    @property
    def device(self):
        return self.dummy_param.device
    
    def attn(self, q, k, v, return_attn=False, prototype_index=None):
        # define some functions
        split = lambda t: einops.rearrange(t, 'b l (h d) -> b h l d', h=self.num_heads) 
        merge = lambda t: einops.rearrange(t, 'b h l d -> b l (h d)')
        norm = lambda t: F.normalize(t, dim=-1)
        
        batch_size = q.size(0)
        num_node = q.size(1)
        
        # full-dot-and-then-exp attention for ablation
        # q, k, v = map(split, [q, k, v])
        # qks = einops.einsum(q, k, 'b h v d, b h V d -> b h v V')
        # output = einops.einsum(F.softmax(qks, dim=-1), v, 'b h v V, b h V d -> b h v d') + v
        # output = merge(output)
               
        # return output
        q, k, v = map(split, [q, k, v])
        q, k = map(norm, [q, k])

        # numerator
        # reduce memory complexity to O(|V|d)
        # reduce time complexity to O(|V|d^2)
        # use v indicates the number of entities, b indicates the batch size, h indicates the number of heads
        # d and D indicate the dimension size, where d == D
        full_rank_term = torch.eye(k.size(-1)).to(self.device)
        full_rank_term = einops.repeat(full_rank_term, 'd D -> b h d D', b=batch_size, h=self.num_heads)
        kvs = einops.einsum(k, v, 'b h v d, b h v D -> b h d D') # torch.cat([einops.einsum(k, v, 'b h v d, b h v D -> b h d D'), full_rank_term], dim=-1)
        numerator =  einops.einsum(q, kvs, 'b h v d, b h d D -> b h v D') # self.fc_attn(einops.einsum(q, kvs, 'b h v d, b h d D -> b h v D'))
        numerator = numerator + einops.reduce(v, 'b h (v w) d -> b h w d', 'sum', w=1) + v*num_node
                    
        # denominator
        # reduce time complexity to O(|V|d)
        denominator = einops.einsum(q, einops.reduce(k, 'b h v d -> b h d', 'sum'), 'b h v d, b h d -> b h v')
        denominator = denominator + torch.full(denominator.shape, fill_value=num_node).to(self.device) + num_node
        denominator = einops.rearrange(denominator, 'b h (v w) -> b h v w', w=1)

        output = numerator / denominator
        output = merge(output)
        
        return output
    def diff_attn(self,q,k,v,return_attn=False,prototype_index=None):
        # define some functions
        
        split = lambda t: einops.rearrange(t, 'b l (h d) -> b h l d', h=self.num_heads) 
        merge = lambda t: einops.rearrange(t, 'b h l d -> b l (h d)')
        norm = lambda t: F.normalize(t, dim=-1)
        
        batch_size = q.size(0)
        num_node = q.size(1)
        
        # full-dot-and-then-exp attention for ablation
        # q, k, v = map(split, [q, k, v])
        # qks = einops.einsum(q, k, 'b h v d, b h V d -> b h v V')
        # output = einops.einsum(F.softmax(qks, dim=-1), v, 'b h v V, b h V d -> b h v d') + v
        # output = merge(output)
               
        # return output
        q, k, v = map(split, [q, k, v])
        q, k = map(norm, [q, k])
        q1,q2=q.chunk(2,dim=-1)
        k1,k2=k.chunk(2,dim=-1)
        #numerator
        kvs_1=einops.einsum(k1,v,'b h v d,b h v D -> b h d D')#b h v d/2,b h v d -> b h d/2 d
        kvs_2=einops.einsum(k2,v,'b h v d,b h v D -> b h d D')#b h v d/2,b h v d -> b h d/2 d
        numerator_1=einops.einsum(q1,kvs_1,'b h v d,b h d D -> b h v D')#b h v d/2,b h d/2 d ->b h v d
        numerator_2=einops.einsum(q2,kvs_2,'b h v d,b h d D -> b h v D')#b h v d/2,b h d/2 d ->b h v d
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        numerator = numerator_1 - lambda_full * numerator_2
        numerator = numerator + einops.reduce(v, 'b h (v w) d -> b h w d', 'sum', w=1) + v*num_node


        '''
        # numerator
        # reduce memory complexity to O(|V|d)
        # reduce time complexity to O(|V|d^2)
        # use v indicates the number of entities, b indicates the batch size, h indicates the number of heads
        # d and D indicate the dimension size, where d == D
        full_rank_term = torch.eye(k.size(-1)).to(self.device)
        full_rank_term = einops.repeat(full_rank_term, 'd D -> b h d D', b=batch_size, h=self.num_heads)
        kvs = einops.einsum(k, v, 'b h v d, b h v D -> b h d D') # torch.cat([einops.einsum(k, v, 'b h v d, b h v D -> b h d D'), full_rank_term], dim=-1)
        numerator =  einops.einsum(q, kvs, 'b h v d, b h d D -> b h v D') # self.fc_attn(einops.einsum(q, kvs, 'b h v d, b h d D -> b h v D'))
        numerator = numerator + einops.reduce(v, 'b h (v w) d -> b h w d', 'sum', w=1) + v*num_node
        '''
        # denominator
        # reduce time complexity to O(|V|d)
        denominator = einops.einsum(q, einops.reduce(k, 'b h v d -> b h d', 'sum'), 'b h v d, b h d -> b h v')
        denominator = denominator + torch.full(denominator.shape, fill_value=num_node).to(self.device) + num_node
        denominator = einops.rearrange(denominator, 'b h (v w) -> b h v w', w=1)

        output = numerator / denominator
        output = merge(output)
        
        return output
    
    def mpnn(self, h_index, r_index, x, z, rev_z, graph, graph_mask, return_attn=False, prototype=None):
        batch_size = x.size(0)
        v_x = torch.zeros(batch_size, graph.num_nodes, self.hidden_dim).to(self.device)
        v_x[torch.arange(batch_size).to(self.device), h_index] = 1
        
        v_x = self.fc_v_x(torch.cat([x, v_x], dim=-1))
        for layer in self.v_layers:
            v_x = layer(v_x, z, r_index, graph, graph_mask)
        return v_x
    def linear_attention(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        stru_weight = 0.2

        x = x + self.attn(q, k, v)
        
        x = self.attn_norm(x)
        x = x + self.ffn(x)
        x = self.norm(x)
        return x
    
        

    # def forward(self, h_index, r_index, x, z, rev_z, graph, graph_mask, return_attn=False, prototype_index=None):
    #     batch_size = x.size(0)

    #     # qk_z = einops.rearrange(self.fc_qk_z(z), 'b (r d) -> b r d', r=self.num_relation)
    #     # qk_x = torch.zeros(batch_size, graph.num_nodes, 1).to(self.device).normal_(0, 4)
    #     # qk_x = self.fc_qk_x(torch.cat([x, qk_x], dim=-1))
    #     # for layer in self.qk_layers:
    #     #     qk_x = layer(qk_x, qk_z, graph, graph_mask)
            
    #     v_x = torch.zeros(batch_size, graph.num_nodes, self.hidden_dim).to(self.device)
    #     v_x[torch.arange(batch_size).to(self.device), h_index] = 1
        
    #     v_x = self.fc_v_x(torch.cat([x, v_x], dim=-1))
    #     for layer in self.v_layers:
    #         v_x = layer(v_x, z, r_index, graph, graph_mask)

    #     # q, k = self.fc_to_qk(qk_x).chunk(2, dim=-1)
    #     # v = v_x 
    #     q = self.W_q(x)
    #     k = self.W_k(x)
    #     v = self.W_v(x)
        
    #     stru_weight = 0.2

    #     x = x + self.attn(q, k, v)
        
    #     x = self.attn_norm(x)
    #     x = x + self.ffn(x)
    #     x = self.norm(x)
    #     x =  (1 - stru_weight) * x + stru_weight * v_x
    #     return x
    
    
class Knowformer(nn.Module):
    def __init__(self, num_relation, num_layer, num_qk_layer, num_v_layer, hidden_dim, num_heads, drop):
        super().__init__()
        self.num_relation = num_relation
        self.num_layer = num_layer
        self.num_qk_layer = num_qk_layer
        self.num_v_layer = num_v_layer
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.drop = drop
        
        # define for getting proper device
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.weight = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))
        #self.weight_ = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))

        self.query_embedding = nn.Embedding(self.num_relation, self.hidden_dim)
        #self.query_embedding_ = nn.Embedding(self.num_relation, self.hidden_dim)
        #layer = KnowformerLayer(self.num_relation, self.num_qk_layer, self.num_v_layer, self.hidden_dim, self.num_heads, self.drop)
        self.layers = nn.ModuleList([deepcopy(KnowformerLayer(_,self.num_relation, self.num_qk_layer, self.num_v_layer, self.hidden_dim, self.num_heads, self.drop)) for _ in range(self.num_layer)])
        #self.layers_ = nn.ModuleList([deepcopy(KnowformerLayer(_,self.num_relation, self.num_qk_layer, self.num_v_layer, self.hidden_dim, self.num_heads, self.drop)) for _ in range(self.num_layer)])
        self.out = nn.Linear(self.hidden_dim , self.hidden_dim)
        
        self.mlp_out = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, 1))
        
    @property
    def device(self):
        return self.dummy_param.device
    
    def forward(self, bacthed_data):
        h_index, r_index, graph, graph_mask = (bacthed_data['h_index'], 
                                               bacthed_data['r_index'], 
                                               bacthed_data['graph'], 
                                               bacthed_data.get('graph_mask', None))
        
        batch_size = h_index.size(0)
        
        rev_r_index = torch.where(r_index % 2 == 1, r_index - 1, r_index + 1)
        
        z = self.query_embedding(r_index)
        #z_ = self.query_embedding_(r_index)
        rev_z = self.query_embedding(rev_r_index).unsqueeze(1)
        #rev_z_ = self.query_embedding_(rev_r_index).unsqueeze(1)
        index = einops.repeat(h_index, 'b -> b v d', v=1, d=self.hidden_dim)
        #x = torch.zeros((batch_size, graph.num_nodes, self.hidden_dim), device=self.device)
        pe = bacthed_data['pe'].unsqueeze(0).repeat(batch_size, 1, 1)
        x = pe
        #x_ = pe
        # x = nn.Embedding(batch_size, graph.num_nodes, self.hidden_dim).to(self.device)
        # x = x.weight

        # for layer,layer_ in zip(self.layers, self.layers_):
        #     # 异步执行
        #     mpnn_future = torch.jit.fork(layer.mpnn, h_index, r_index, x, z, rev_z, graph, graph_mask)
        #     #mpnn_future_ = torch.jit.fork(layer_.mpnn, h_index, r_index, x_, z_, rev_z_, graph, graph_mask)
        #     l_a_future = torch.jit.fork(layer.linear_attention, x)
        #     #l_a_future_ = torch.jit.fork(layer_.linear_attention,x_)
        #     # 等待任务完成
        #     local = torch.jit.wait(mpnn_future)
        #     #local_ = torch.jit.wait(mpnn_future_)
        #     glob = torch.jit.wait(l_a_future)
        #     #glob_ = torch.jit.wait(l_a_future_)
        #     weight = torch.sigmoid(self.weight)
        #     #weight_ = torch.sigmoid(self.weight_)
        #     x = weight * local + (1 - weight) * glob
        #     #x_ =weight_ * local_ + (1 - weight) * glob_
        for layer in self.layers:
            mpnn_future = torch.jit.fork(layer.mpnn, h_index, r_index, x, z, rev_z, graph, graph_mask)
            l_a_future = torch.jit.fork(layer.linear_attention, x)
            local = torch.jit.wait(mpnn_future)
            glob = torch.jit.wait(l_a_future)
            weight = torch.sigmoid(self.weight)
            x = weight * local + (1 - weight) * glob
        #feat = torch.cat((x, x_), dim=-1)
        feat = x
        feat = self.out(feat)
        score = self.mlp_out(feat).squeeze(-1)
        
        return score


def create_projection_matrix(m, d, seed=0, scaling=0, struct_mode=False):
    nb_full_blocks = int(m/d)
    block_list = []
    current_seed = seed.item()
    for _ in range(nb_full_blocks):
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d), generator=torch.Generator().manual_seed(current_seed)) * math.sqrt(2)
            q, _ = torch.linalg.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q)
        current_seed += 1
    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d), generator=torch.Generator().manual_seed(current_seed)) * math.sqrt(2)
            q, _ = torch.linalg.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q[0: remaining_rows])
    final_matrix = torch.vstack(block_list)

    current_seed += 1
    if scaling == 0:
        multiplier = torch.norm(torch.randn((m, d), generator=torch.Generator().manual_seed(current_seed)), dim=1) * math.sqrt(2)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
    else:
        raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

    return torch.matmul(torch.diag(multiplier), final_matrix)

def create_products_of_givens_rotations(dim, seed):
    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim, dim)
    rng = np.random.default_rng(seed)
    for _ in range(nb_givens_rotations):
        random_angle = math.pi * rng.uniform()
        random_indices = rng.choice(dim, 2)
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i = q[index_i]
        slice_j = q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + math.cos(random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return torch.tensor(q, dtype=torch.float32)