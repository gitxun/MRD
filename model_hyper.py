import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv, FAConv
from torch.nn import Parameter
import numpy as np, itertools, random, copy, math
import math
import scipy.sparse as sp
from model_GCN import GCNII_lyc
import ipdb
from HypergraphConv import HypergraphConv
from torch_geometric.nn import GCNConv
from itertools import permutations
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops
from high_fre_conv import highConv
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

# class GatedModalFusion(nn.Module):
#     def __init__(self, feature_dim):
#         super(GatedModalFusion, self).__init__()
#         self.feature_dim = feature_dim
#         # 门控生成器：生成 gate 值
#         self.gate_l = nn.Linear(feature_dim, feature_dim)
#         self.gate_a = nn.Linear(feature_dim, feature_dim)
#         self.gate_v = nn.Linear(feature_dim, feature_dim)

#     def forward(self, l, a, v):
#         """
#         l, a, v: 分别是 Length-based、Acoustic 和 Visual 特征
#                  每个的形状为 [对话长度, 特征维度]
#         """
#         # 计算每个模态的门控值
#         g_l = torch.sigmoid(self.gate_l(l))  # [对话长度, 特征维度]
#         g_a = torch.sigmoid(self.gate_a(a))
#         g_v = torch.sigmoid(self.gate_v(v))
        
#         # 门控调制模态特征
#         l_gated = l * g_l # [对话长度, 特征维度]
#         a_gated = a * g_a
#         v_gated = v * g_v
        
#         # 拼接门控后的特征
#         fused_feature = torch.cat([l_gated, a_gated, v_gated], dim=-1)  # [对话长度, 3 * 特征维度]
#         return fused_feature

class GatedModalFusion(nn.Module):
    def __init__(self, feature_dim):
        super(GatedModalFusion, self).__init__()
        self.feature_dim = feature_dim

        # 门控生成器：使用小型 MLP
        self.gate_l = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(feature_dim, feature_dim)
        )
        self.gate_a = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, feature_dim)
        )
        self.gate_v = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(feature_dim, feature_dim)
        )

        # 可学习模态贡献权重
        self.alpha_l = nn.Parameter(torch.tensor(0.6))
        self.alpha_a = nn.Parameter(torch.tensor(0.3))  
        self.alpha_v = nn.Parameter(torch.tensor(0.1))  


    def forward(self, l, a, v):
        # 计算每个模态的门控值
        g_l = torch.sigmoid(self.gate_l(l))
        g_a = torch.sigmoid(self.gate_a(a))
        g_v = torch.sigmoid(self.gate_v(v))

        # 门控调制模态特征 + 残差连接
        l_gated = l * g_l + l
        a_gated = a * g_a + a
        v_gated = v * g_v + v
        fused_l = l_gated*self.alpha_l + l
        fused_a = a_gated*self.alpha_a + a
        fused_v = v_gated*self.alpha_v + v
        # 动态加权融合
        fused_feature = torch.cat([fused_l, fused_a, fused_v], dim=-1)  # 在最后一个维度上拼接
        return fused_feature


class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim, num_heads=1):
        super(CrossAttentionFusion, self).__init__()
        
        # 定义独立的 Query、Key、Value 投影
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        # 定义视觉模态的动态权重分配（小型 MLP）
        self.visual_weight_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(feature_dim // 2, 1),  # 输出一个标量权重
            nn.Sigmoid()  # 将权重限制在 [0, 1] 范围内
        )
        
        # 动态的模态权重
        self.alpha_a = nn.Parameter(torch.tensor(1.0))  # 音频
        self.alpha_l = nn.Parameter(torch.tensor(1.0))  # 文本
        
        # 定义多头注意力的设置
        self.num_heads = num_heads
        self.attention_head_dim = feature_dim // num_heads
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, a, v, l):
        # 生成 Query/Key/Value
        query_l = self.query_proj(l)  # 文本的 Query
        key_a = self.key_proj(a)      # 音频的 Key
        value_a = self.value_proj(a)  # 音频的 Value
        
        # 文本-query 和 音频-key
        attn_l_to_a = torch.matmul(query_l, key_a.transpose(-2, -1)) / self.attention_head_dim ** 0.5
        attn_l_to_a = self.softmax(attn_l_to_a)
        l_fused_a = torch.matmul(attn_l_to_a, value_a)
        
        # 加入自注意力机制
        l_self = l  # 文本的自注意特征
        a_self = a  # 音频的自注意特征
        
        # 动态模态加权（文本为主模态，音频为次要模态）
        fused_l = torch.sigmoid(self.alpha_l) * l_fused_a + (1 - torch.sigmoid(self.alpha_l)) * l_self
        fused_a = torch.sigmoid(self.alpha_a) * a_self  # 音频的自注意增强
        
        # 视觉模态的动态权重生成
        visual_weight = self.visual_weight_mlp(v)  # 生成动态权重（标量）
        fused_v = visual_weight * v  # 重新加权的视觉模态特征
        
        # 保留输入维度不变
        fused_l += l
        fused_a += a
        
        # 拼接融合后的特征，维度为 [对话长度, 3 * 特征维度]
        fused_feature = torch.cat([fused_l, fused_a, fused_v], dim=-1)
        
        # 输出融合后的特征
        return fused_feature

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x, dia_len):
        """
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        tmpx = torch.zeros(0).cuda()
        tmp = 0
        for i in dia_len:
            a = x[tmp:tmp+i].unsqueeze(1)
            a = a + self.pe[:a.size(0)]
            tmpx = torch.cat([tmpx,a], dim=0)
            tmp = tmp+i
        #x = x + self.pe[:x.size(0)]
        tmpx = tmpx.squeeze(1)
        return self.dropout(tmpx)

class HyperGCN(nn.Module):
    def __init__(self, a_dim, v_dim, l_dim, n_dim, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, return_feature, use_residue, 
                new_graph='full',n_speakers=2, modals=['a','v','l'], use_speaker=True, 
                use_modal=False, num_L=3, num_K=4):
        super(HyperGCN, self).__init__()
        
        # 加入控融合模块
        # self.gated_fusion = GatedModalFusion(feature_dim=n_dim)  # n_dim 是单一模态特征的维度
       
        # self.cross_attention_fusion = CrossAttentionFusion(feature_dim=n_dim, num_heads=2)
        
        self.return_feature = return_feature  #True
        self.use_residue = use_residue
        self.new_graph = new_graph
    

        #self.graph_net = GCNII_lyc(nfeat=n_dim, nlayers=nlayers, nhidden=nhidden, nclass=nclass,
        #                       dropout=dropout, lamda=lamda, alpha=alpha, variant=variant,
        #                       return_feature=return_feature, use_residue=use_residue)
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.use_position = False
        
        #------------------------------------    
        self.fc1 = nn.Linear(n_dim, nhidden)      
        #self.fc2 = nn.Linear(n_dim, nhidden)     
        self.num_L =  num_L
        self.num_K =  num_K
        for ll in range(num_L):
            setattr(self,'hyperconv%d' %(ll+1), HypergraphConv(nhidden, nhidden))
        self.act_fn = nn.ReLU()
        self.hyperedge_weight = nn.Parameter(torch.ones(1000))
        self.EW_weight = nn.Parameter(torch.ones(5200))
        self.hyperedge_attr1 = nn.Parameter(torch.rand(nhidden))
        self.hyperedge_attr2 = nn.Parameter(torch.rand(nhidden))
        #nn.init.xavier_uniform_(self.hyperedge_attr1)
        for kk in range(num_K):
            setattr(self,'conv%d' %(kk+1), highConv(nhidden, nhidden))
        #self.conv = highConv(nhidden, nhidden)

    def forward(self, a, v, l, dia_len, qmask, epoch, mask_context=None):
        qmask = torch.cat([qmask[:x,i,:] for i,x in enumerate(dia_len)],dim=0)
        spk_idx = torch.argmax(qmask, dim=-1)
        spk_emb_vector = self.speaker_embeddings(spk_idx)
        if self.use_speaker:
            if 'l' in self.modals:
                l += spk_emb_vector
        if self.use_position:
            if 'l' in self.modals:
                l = self.l_pos(l, dia_len)
            if 'a' in self.modals:
                a = self.a_pos(a, dia_len)
            if 'v' in self.modals:
                v = self.v_pos(v, dia_len)
        if self.use_modal:  
            emb_idx = torch.LongTensor([0, 1, 2]).cuda()
            emb_vector = self.modal_embeddings(emb_idx)

            if 'a' in self.modals:
                a += emb_vector[0].reshape(1, -1).expand(a.shape[0], a.shape[1])
            if 'v' in self.modals:
                v += emb_vector[1].reshape(1, -1).expand(v.shape[0], v.shape[1])
            if 'l' in self.modals:
                l += emb_vector[2].reshape(1, -1).expand(l.shape[0], l.shape[1])

        # fused_features = torch.cat((l, l, l), dim=-1)
        # a, v, l = torch.chunk(fused_features, chunks=3, dim=-1)
        hyperedge_index, edge_index, features, batch, hyperedge_type1 = self.create_hyper_index(a, v, l, dia_len, self.modals)
        x1 = self.fc1(features)  
        weight = self.hyperedge_weight[0:hyperedge_index[1].max().item()+1]
        EW_weight = self.EW_weight[0:hyperedge_index.size(1)]

        edge_attr = self.hyperedge_attr1*hyperedge_type1 + self.hyperedge_attr2*(1-hyperedge_type1)
        out = x1
        for ll in range(self.num_L):
            out = getattr(self,'hyperconv%d' %(ll+1))(out, hyperedge_index, weight, edge_attr, EW_weight, dia_len)             
        if self.use_residue:
            out1 = torch.cat([features, out], dim=-1)                                   
        #out1 = self.reverse_features(dia_len, out1)                                     

        #---------------------------------------
        gnn_edge_index, gnn_features = self.create_gnn_index(a, v, l, dia_len, self.modals, mask_context, qmask)
        gnn_out = x1
        for kk in range(self.num_K):
            gnn_out = gnn_out + getattr(self,'conv%d' %(kk+1))(gnn_out,gnn_edge_index)

        out2 = torch.cat([out,gnn_out], dim=1)
        if self.use_residue:
            out2 = torch.cat([features, out2], dim=-1)
        out1 = self.reverse_features(dia_len, out2)
        
        #---------------------------------------
        return out1
        


    def create_hyper_index(self, a, v, l, dia_len, modals):
        self_loop = False
        num_modality = len(modals)
        node_count = 0
        edge_count = 0
        batch_count = 0
        index1 =[]
        index2 =[]
        tmp = []
        batch = []
        edge_type = torch.zeros(0).cuda()
        in_index0 = torch.zeros(0).cuda()
        hyperedge_type1 = []
        for i in dia_len:
            nodes = list(range(i*num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i*num_modality//3]
            nodes_a = nodes[i*num_modality//3:i*num_modality*2//3]
            nodes_v = nodes[i*num_modality*2//3:]
            index1 = index1 + nodes_l + nodes_a + nodes_v
            for _ in range(i):
                index1 = index1 + [nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]]
            for _ in range(i+3):
                if _ < 3:
                    index2 = index2 + [edge_count]*i
                else:
                    index2 = index2 + [edge_count]*3
                edge_count = edge_count + 1
            if node_count == 0:
                ll = l[0:0+i]
                aa = a[0:0+i]
                vv = v[0:0+i]
                features = torch.cat([ll,aa,vv],dim=0)
                temp = 0+i
            else:
                ll = l[temp:temp+i]
                aa = a[temp:temp+i]
                vv = v[temp:temp+i]

                features_temp = torch.cat([ll,aa,vv],dim=0)
                features =  torch.cat([features,features_temp],dim=0)
                temp = temp+i
            
            Gnodes=[]
            Gnodes.append(nodes_l)
            Gnodes.append(nodes_a)
            Gnodes.append(nodes_v)
            for _ in range(i):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
            for ii, _ in enumerate(Gnodes):
                perm = list(permutations(_,2))
                tmp = tmp + perm
            batch = batch + [batch_count]*i*3
            batch_count = batch_count + 1
            hyperedge_type1 = hyperedge_type1 + [1]*i + [0]*3

            node_count = node_count + i*num_modality

        index1 = torch.LongTensor(index1).view(1,-1)
        index2 = torch.LongTensor(index2).view(1,-1)
        hyperedge_index = torch.cat([index1,index2],dim=0).cuda() 
        if self_loop:
            max_edge = hyperedge_index[1].max()
            max_node = hyperedge_index[0].max()
            loops = torch.cat([torch.arange(0,max_node+1,1).repeat_interleave(2).view(1,-1),
                            torch.arange(max_edge+1,max_edge+1+max_node+1,1).repeat_interleave(2).view(1,-1)],dim=0).cuda()
            hyperedge_index = torch.cat([hyperedge_index, loops], dim=1)

        edge_index = torch.LongTensor(tmp).T.cuda()
        batch = torch.LongTensor(batch).cuda()

        hyperedge_type1 = torch.LongTensor(hyperedge_type1).view(-1,1).cuda()

        return hyperedge_index, edge_index, features, batch, hyperedge_type1
    
    def reverse_features(self, dia_len, features):
        l = []
        a = []
        v = []
        for i in dia_len:
            ll = features[0:1 * i]
            aa = features[1 * i:2 * i]
            vv = features[2 * i:3 * i]
            features = features[3 * i:]
            
            # # 打印每种模态特征的分布
            # print(f"Length-based feature (L): {ll.shape}")
            # print(f"Acoustic feature (A): {aa.shape}")
            # print(f"Visual feature (V): {vv.shape}")
            
            l.append(ll)
            a.append(aa)
            v.append(vv)
        
        tmpl = torch.cat(l, dim=0)
        tmpa = torch.cat(a, dim=0)
        tmpv = torch.cat(v, dim=0)
        
        # # 打印拼接后的模态特征
        # print(f"Concatenated Length-based features (tmpl): {tmpl.shape}")
        # print(f"Concatenated Acoustic features (tmpa): {tmpa.shape}")
        # print(f"Concatenated Visual features (tmpv): {tmpv.shape}")
        
        features = torch.cat([tmpl, tmpa, tmpv], dim=-1)
        
        # # 打印最终拼接后的特征
        # print(f"Final combined features: {features.shape}")
        
        return features



    def create_gnn_index(self, a, v, l, dia_len, modals, mask_context=None, qmask=None):
        self_loop = False
        num_modality = len(modals)
        node_count = 0
        batch_count = 0
        index =[]
        tmp = []
        
        for i in dia_len:
            nodes = list(range(i*num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i*num_modality//3]
            nodes_a = nodes[i*num_modality//3:i*num_modality*2//3]
            nodes_v = nodes[i*num_modality*2//3:]
            index = index + list(permutations(nodes_l,2)) + list(permutations(nodes_a,2)) + list(permutations(nodes_v,2))
            Gnodes=[]
            for _ in range(i):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
            for ii, _ in enumerate(Gnodes):
                tmp = tmp +  list(permutations(_,2))
            if node_count == 0:
                ll = l[0:0+i]
                aa = a[0:0+i]
                vv = v[0:0+i]
                features = torch.cat([ll,aa,vv],dim=0)
                temp = 0+i
            else:
                ll = l[temp:temp+i]
                aa = a[temp:temp+i]
                vv = v[temp:temp+i]
                features_temp = torch.cat([ll,aa,vv],dim=0)
                features =  torch.cat([features,features_temp],dim=0)
                temp = temp+i
            node_count = node_count + i*num_modality
        

        if mask_context == 'intra':
            ## To remove intra speaker edges
            speaker_index = torch.nonzero(qmask)[:,1].tolist()
            new_index = []
            mask_index_count = 0
            mask_node_count = 0
            for cur_len in dia_len:
                dia_speaker = speaker_index[mask_node_count:mask_node_count+cur_len] * 3
                dia_speaker = {i+mask_node_count*3:dia_speaker[i] for i in range(len(dia_speaker))}
                for cur_index in index[mask_index_count : mask_index_count + (cur_len -1) * cur_len * 3]:
                    if dia_speaker[cur_index[0]] != dia_speaker[cur_index[1]]:
                        new_index.append(cur_index)
                mask_node_count += cur_len
                mask_index_count = (mask_index_count + (cur_len -1) * cur_len * 3)
            index = new_index
        elif mask_context == 'inter':
            speaker_index = torch.nonzero(qmask)[:,1].tolist()
            new_index = []
            mask_index_count = 0
            mask_node_count = 0
            for cur_len in dia_len:
                dia_speaker = speaker_index[mask_node_count:mask_node_count+cur_len] * 3
                dia_speaker = {i+mask_node_count*3:dia_speaker[i] for i in range(len(dia_speaker))}
                for cur_index in index[mask_index_count : mask_index_count + (cur_len -1) * cur_len * 3]:
                    if dia_speaker[cur_index[0]] == dia_speaker[cur_index[1]]:
                        new_index.append(cur_index)
                mask_node_count += cur_len
                mask_index_count = (mask_index_count + (cur_len -1) * cur_len * 3)
            index = new_index
            
        edge_index = torch.cat([torch.LongTensor(index).T,torch.LongTensor(tmp).T],1).cuda()
        return edge_index, features