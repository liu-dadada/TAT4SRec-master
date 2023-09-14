import numpy as np
import torch
from torch import nn as nn
import math
import world
import torch.nn.functional as F


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
class LightGCN(BasicModel):  # 它继承自之前提到的 BasicModel 类。
    def __init__(self,
                 config: dict,
                 dataset):
        # 它接收两个参数，config 和 dataset，分别表示配置信息（一个字典）和数据集（一个 BasicDataset 类的实例）。
        super(LightGCN, self).__init__()  # 调用了父类 BasicModel 的构造函数，确保基类的初始化操作得以执行
        self.config = config
        self.dataset = dataset  # 这行代码将传入的数据集实例保存在 LightGCN 类的属性 dataset 中，并指定数据集的类型为 dataloader.BasicDataset。
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.username
        self.num_items = self.dataset.itemname
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        # #self.num_users、self.num_items、self.latent_dim 等行代码将数据集中的用户数、物品数、潜在特征维度等基础信息保存到模型中
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        # 使用 torch.nn.Embedding 创建用户和物品的嵌入层。这将把用户和物品的 ID 映射到一个指定维度的向量表示。
        if self.config['pretrain'] == 0:
            #             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            #             print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
            # 如果 self.config['pretrain'] 为 0，表示不使用预训练数据，那么将使用正态分布来初始化嵌入层的权重。这里采用的是正态分布初始化，通过 nn.init.normal_ 函数。
            # 否则，将使用预训练的数据来初始化权重，将预训练的用户和物品嵌入数据赋值给相应的权重。
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        # 获取了稀疏图的信息，它将在模型中用于图卷积操作。
        print(f"lgn is already to go(dropout:{self.config['dropout']})")
        # 输出了一些信息，例如模型已准备好进行训练，以及使用的参数设置（如 dropout）。
        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        # 稀疏张量 x 进行 dropout 操作，按照给定的 keep_prob 概率保留元素。
        size = x.size()  # 获取稀疏张量 x 的形状。
        index = x.indices().t()  # 获取稀疏张量 x 的索引，使用 .t() 对索引进行转置。
        values = x.values()  # 获取稀疏张量 x 的值。
        random_index = torch.rand(len(values)) + keep_prob  # 生成一个与 values 长度相同的随机张量，表示保留元素的概率。这里将 keep_prob 加到随机值上。
        random_index = random_index.int().bool()  # 将上一步生成的随机张量转换为整数，然后将其转换为布尔型张量。这样就得到了一个与 values 长度相同的布尔型张量，表示每个元素是否被保留。
        index = index[random_index]
        values = values[random_index] / keep_prob  # 根据生成的布尔型张量，将保留的元素的索引和值提取出来，并将值除以 keep_prob 进行归一化。
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
        # 使用提取出的索引、值和形状，创建一个新的稀疏张量 g。

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        # 将用户和物品的嵌入权重连接在一起，得到一个张量 all_emb，用于表示所有实体的嵌入。
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph
            # 如果是训练状态，调用私有方法 __dropout 对图进行 dropout 操作。
        # 如果不是训练状态，保持图不变。
        for layer in range(self.n_layers):  # 循环遍历图卷积的层数。
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # 将 embs 列表中的嵌入向量堆叠起来，构成一个多维张量，表示不同层的嵌入向量。
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        # 计算所有层嵌入向量的平均值，得到一个平均的嵌入向量 light_out。
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        # 平均的嵌入向量按照用户和物品进行分割，得到用户嵌入向量 users 和物品嵌入向量 items。
        return users, items
        # 最后返回计算得到的用户和物品嵌入向量。

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        # 方法计算得到所有用户和物品的嵌入向量。
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        # 从计算得到的用户和物品嵌入向量中，分别选取给定用户、正样本物品和负样本物品的嵌入向量。
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        # 分别从用户和物品的嵌入层中获取给定用户、正样本物品和负样本物品的 "ego"（自我）嵌入向量。这些向量是未经过计算的原始嵌入向量。
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
        # 一次性获取多种嵌入向量，包括计算得到的、未经过计算的原始嵌入向量，以及用户、正样本物品和负样本物品的不同类型的嵌入向量

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class DiscreteEmbedding(nn.Module):
    def __init__(self,output_dims,num_points,device='cuda'):
        super(DiscreteEmbedding, self).__init__()
        self.output_dims = output_dims
        self.num_points = num_points

        self.time_embedding = nn.Parameter(torch.Tensor(self.num_points,self.output_dims))
        #创建一个可学习的参数self.time_embedding，形状为(num_points, output_dims)，用于表示时间嵌入向量。
        self.points = torch.arange(0,self.num_points).to(device)
        #self.time_embedding = nn.Embedding(self.num_points,self.output_dims)
        #self.one_hot_embedding = torch.Tensor(torch.zeros([x.shape[0],x.shape[1],self.num_points]))

    # def _rect_window(self,x,window_size = 8):
    #     w_2 = window_size / 2
    #     return (torch.sign(x+w_2)-torch.sign(x-w_2))/2
    def _rect_window(self, x, window_size=8):
        rect_value = (torch.sign(x + window_size / 2) - torch.sign(x - window_size / 2)) / 2 * (
                1 + torch.sign(x + window_size / 2)) * (1 - torch.sign(x - window_size / 2))
        return rect_value

    def forward(self,x): # [N,L]

        x *= self.num_points
        x = x.unsqueeze(-1)
        w = x - self.points
        w = self._rect_window(w, window_size=1)

        output = torch.matmul(w, self.time_embedding)
        return output







class ContinuousEmbedding(nn.Module):
    def __init__(self,output_dims,num_points,minval=-1.0,maxval=1.0,window_size = 8,window_type='hann',normalized = True,device='cpu'):
        super(ContinuousEmbedding,self).__init__()
        self.output_dims = output_dims
        self.minval = minval
        self.maxval = maxval
        self.num_points = num_points
        self.window_size = window_size
        assert window_type in {'triangular', 'rectangular', 'hann'}
        self.window_type = window_type
        self.normalized = normalized

        self.embedding = nn.Parameter(torch.Tensor(self.num_points,self.output_dims))
        self.embedding_dim = self.output_dims

        if self.window_type == 'hann':
            self.window_func = self._hann_window
        elif self.window_type == 'triangular':
            self.window_func = self._triangle_window
        else:
            self.window_func = self._rect_window

        self.points = torch.arange(0,self.num_points).to(device)

    def _rect_window(self,x, window_size=8):
        rect_value = (torch.sign(x + window_size / 2) - torch.sign(x - window_size / 2)) / 2 * (
                    1 + torch.sign(x + window_size / 2)) * (1 - torch.sign(x - window_size / 2))
        return rect_value
    # def _rect_window(self,x,window_size = 8):
    #     w_2 = window_size / 2
    #     return (torch.sign(x+w_2)-torch.sign(x-w_2))/2
    #
    # def _triangle_window(self, x, window_size=16):
    #     w_2 = window_size / 2
    #     return (torch.abs(x + w_2) + torch.abs(x - w_2) - 2 * torch.abs(x)) / window_size

    def _triangle_window(self,x, window_size= 16):
        rect_value = (torch.sign(x + window_size / 2) - torch.sign(x - window_size / 2)) / 2 * (
                    1 + torch.sign(x + window_size / 2)) * (1 - torch.sign(x - window_size / 2))
        triangle_value = (torch.abs(x + window_size / 2) + torch.abs(x - window_size / 2) - 2 * torch.abs(
            x)) / window_size
        combined_value = rect_value + triangle_value
        return combined_value
    def _hann_window(self, x, window_size=16):
        y = torch.cos(math.pi * x / window_size)
        y = y * y * self._rect_window(x, window_size=window_size)
        return y

    def forward(self,x): # x:[N,L]
        x -= self.minval
        x *= self.num_points/(self.maxval-self.minval)
        x = x.unsqueeze(-1)
        #print('x.type',type(x))
        #print('point.type',type(self.points))
        w = x-self.points
        w = self.window_func(w,window_size=self.window_size)
        if self.normalized:

            w = w/w.sum(-1,keepdim=True)

        output = torch.matmul(w,self.embedding)
        return output


class Encoder_layer(nn.Module):
    def __init__(self,block_nums,hidden_units,head_num,dropout_rate,if_point_wise=False,if_gelu=False):
        super(Encoder_layer, self).__init__()

        self.block_nums = block_nums
        self.hidden_units = hidden_units
        self.head_num = head_num
        self.dropout_rate = dropout_rate
        self.if_point_wise = if_point_wise
        self.if_gelu = if_gelu

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)

        for _ in range(self.block_nums):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_units,eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            #print(self.hidden_units,self.head_num)
            new_attn_layer = torch.nn.MultiheadAttention(self.hidden_units,
                                                         self.head_num,
                                                         self.dropout_rate)

            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layer = PointWiseFeedForward(self.hidden_units, self.dropout_rate,self.if_point_wise,self.if_gelu)
            self.forward_layers.append(new_fwd_layer)

    class Encoder_layer(nn.Module):
        def __init__(self, block_nums, hidden_units, head_num, dropout_rate, if_point_wise=False, if_gelu=False):
            super(Encoder_layer, self).__init__()

            self.block_nums = block_nums
            self.hidden_units = hidden_units
            self.head_num = head_num
            self.dropout_rate = dropout_rate
            self.if_point_wise = if_point_wise
            self.if_gelu = if_gelu

            self.attention_layernorms = torch.nn.ModuleList()
            self.attention_layers = torch.nn.ModuleList()
            self.forward_layernorms = torch.nn.ModuleList()
            self.forward_layers = torch.nn.ModuleList()
            self.last_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)

            for _ in range(self.block_nums):
                new_attn_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
                self.attention_layernorms.append(new_attn_layernorm)

                new_attn_layer = torch.nn.MultiheadAttention(self.hidden_units,
                                                             self.head_num,
                                                             self.dropout_rate)
                self.attention_layers.append(new_attn_layer)

                new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
                self.forward_layernorms.append(new_fwd_layernorm)

                new_fwd_layer = PointWiseFeedForward(self.hidden_units, self.dropout_rate, self.if_point_wise,
                                                     self.if_gelu)
                self.forward_layers.append(new_fwd_layer)

        def forward(self, seqs, attn_mask=None, key_padding_mask=None):
            for i in range(self.block_nums):
                seqs = torch.transpose(seqs, 0, 1)
                Q = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attn_mask)
                seqs = Q + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)

                seqs = self.forward_layernorms[i](seqs)
                seqs = self.forward_layers[i](seqs)
                seqs *= ~key_padding_mask.unsqueeze(-1)

            output = self.last_layernorm(seqs)
            return output

    def forward(self,seqs,attn_mask=None,key_padding_mask=None):
        for i in range(self.block_nums):
            seqs = torch.transpose(seqs,0,1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs,_ = self.attention_layers[i](Q,seqs,seqs,
                                                     attn_mask = attn_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0 ,1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~key_padding_mask.unsqueeze(-1)

        output = self.last_layernorm(seqs)
        return output

class Decoder_layer(nn.Module):
    def __init__(self, hidden_units, head_num, block_num,dropout_rate,if_point_wise=False,if_gelu=False):
        super(Decoder_layer, self).__init__()
        #self.block_nums = block_nums
        self.hidden_units = hidden_units
        self.head_num = head_num
        self.dropout_rate = dropout_rate
        self.block_num = block_num
        self.if_point_wise = if_point_wise
        self.if_gelu = if_gelu

        #self.attention_layernorms1 = nn.LayerNorm(self.hidden_units, eps=1e-8)
        #self.attention_layernorms2 = nn.LayerNorm(self.hidden_units, eps=1e-8)
        #self.self_attention_layers = nn.MultiheadAttention(self.hidden_units,
        #                                                 self.head_num,
        #                                                 self.dropout_rate)
        #self.mutil_attention_layers = nn.MultiheadAttention(self.hidden_units,
        #                                                 self.head_num,
        #                                                 self.dropout_rate)
        #self.forward_layers = PointWiseFeedForward(self.hidden_units, self.dropout_rate)
        #self.dropout1 = nn.Dropout(self.dropout_rate)
        #self.dropout2 = nn.Dropout(self.dropout_rate)
        #self.last_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)

        self.attention_layernorms1 = nn.ModuleList()
        self.attention_layernorms2 = nn.ModuleList()
        self.self_attention_layers = nn.ModuleList()
        self.mutil_attention_layers = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.dropout1 = nn.ModuleList()
        self.dropout2 = nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)

        for _ in range(self.block_num):
            new_attn_layernorm1 = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.attention_layernorms1.append(new_attn_layernorm1)
            new_attn_layernorm2 = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.attention_layernorms2.append(new_attn_layernorm2)

            self_attn_layer = torch.nn.MultiheadAttention(self.hidden_units,
                                                         self.head_num,
                                                         self.dropout_rate)
            self.self_attention_layers.append(self_attn_layer)

            self.dropout1.append(nn.Dropout(self.dropout_rate))
            self.dropout2.append(nn.Dropout(self.dropout_rate))
            mutil_attention_layers = torch.nn.MultiheadAttention(self.hidden_units,
                                                         self.head_num,
                                                         self.dropout_rate)
            self.mutil_attention_layers.append(mutil_attention_layers)

            new_fwd_layer = PointWiseFeedForward(self.hidden_units, self.dropout_rate,self.if_point_wise,self.if_gelu)
            self.forward_layers.append(new_fwd_layer)

    def forward(self,tgt,memory,attn_mask=None,key_padding_mask=None,memory_mask=None):
        memory = torch.transpose(memory, 0, 1)
        for i in range(self.block_num):
            tgt = torch.transpose(tgt, 0, 1)
            #Q = self.attention_layernorms(tgt)
            mha_outputs, _ = self.self_attention_layers[i](tgt, tgt, tgt,
                                                  attn_mask=attn_mask)
            tgt = tgt + self.dropout1[i](mha_outputs)

            tgt =  self.attention_layernorms1[i](tgt)
            if key_padding_mask is not None:
                tgt = torch.transpose(tgt, 0, 1)
                tgt *= ~key_padding_mask.unsqueeze(-1)
                tgt = torch.transpose(tgt,0, 1)

            tgt2,_ = self.mutil_attention_layers[i](tgt,memory,memory,attn_mask=memory_mask)
            #tgt2, _ = self.mutil_attention_layers[i](tgt, memory, memory,attn_mask=None)
            tgt = tgt + self.dropout2[i](tgt2)
            tgt = torch.transpose(tgt, 0, 1)
            tgt = self.attention_layernorms2[i](tgt)
            tgt = self.forward_layers[i](tgt)
            tgt *= ~key_padding_mask.unsqueeze(-1)
        tgt = self.last_layernorm(tgt)
        return tgt



class TAT4Rec(nn.Module):
    def __init__(self,user_num,item_num,args):
        super(TAT4Rec,self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.lag_from_now_emb = ContinuousEmbedding(output_dims=args.hidden_units,
                                                    num_points=args.lag_time_bins,
                                                    minval=0.0, maxval=1.0,
                                                    window_size=args.lagtime_window_size,
                                                    window_type='hann',
                                                    normalized=True,
                                                    device=args.device
                                                    )
        #self.discrete_timeembedding = DiscreteEmbedding(output_dims=args.hidden_units,num_points=args.lag_time_bins,device=args.device)


        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units_item)
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units_item, padding_idx=0)
        self.encoder = Encoder_layer(args.encoder_blocks_num,args.hidden_units,args.encoder_heads_num,args.dropout_rate_encoder,args.if_point_wise_feedforward)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate_decoder)

        self.decoder = Decoder_layer(args.hidden_units, args.decoder_heads_num, args.decoder_blocks_num,args.dropout_rate_decoder,args.if_point_wise_feedforward)
        self.max_len = args.maxlen


    def log2feats(self, user_ids,log_seqs,seq_ln,max_time_lag):

        item = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        #print('item_shape1:', item.shape)
        item *= self.item_emb.embedding_dim ** 0.5

        positions = np.tile(np.array(range(item.shape[1])), [item.shape[0], 1])
        item += self.pos_emb(torch.LongTensor(positions).to(self.dev))  # 加上postional embedding [N,L,E]
        item = self.emb_dropout(item)

        seq_ln = torch.FloatTensor(seq_ln).to(self.dev)
        #seq_ln = torch.FloatTensor(seq_ln)
        seq_ln = torch.clamp(seq_ln,min=0,max=max_time_lag)
        seq_ln = seq_ln/max_time_lag
        #seq_ln = torch.LongTensor(seq_ln).to(self.dev)
        seq_ln = self.lag_from_now_emb(seq_ln)  # [N,L,E]
        #seq_ln = self.discrete_timeembedding(seq_ln)
        #print('seqln:',seq_ln.shape)
        #if self.if_scale:
        #    seq_ln *= self.lag_from_now_emb.embedding_dim ** 0.5

        padding_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)  # [N,L]
        tl = item.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))  # [L,L]
        encoder_output = self.encoder(seq_ln,attn_mask = attention_mask,key_padding_mask =padding_mask )
        decoder_output = self.decoder(item,encoder_output,attn_mask= attention_mask,key_padding_mask=padding_mask,memory_mask=attention_mask)
        return decoder_output

    def forward(self, user_ids, log_seqs, seq_ts,seq_ln,pos_seqs, neg_seqs,x_mask,max_time_lag):
        o = self.log2feats(user_ids,log_seqs,seq_ln,max_time_lag)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))  # 目标 (N,L,E)
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))  # 负采样 (N,L,E)

        pos_logits = (o * pos_embs).sum(dim=-1)
        neg_logits = (o * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, seq_ln,item_indices,max_time_lag): # for inference
        log_feats = self.log2feats(user_ids,log_seqs,seq_ln,max_time_lag) # user_ids hasn't been used yet [1,L,E]
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste [1,E]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # [101,E]
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1) # [1,101]

        return logits # preds # (U, I)




class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate,if_point_wise=False,if_gelu=False):

        super(PointWiseFeedForward, self).__init__()
        self.if_point_wise = if_point_wise
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.if_gelu = if_gelu
        if if_gelu:
            self.act = torch.nn.GELU()
        else:
            self.act = torch.nn.ReLU()
        #self.relu = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)
        if  self.if_point_wise:
            self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
            self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        else:
            self.feedforward1 = torch.nn.Linear(hidden_units,hidden_units)
            self.feedforward2 = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, inputs):
        if self.if_point_wise:
            outputs = self.dropout2(self.conv2(self.act(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
            outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
            outputs += inputs # 注意此处有残差机制
        else:
            outputs = self.dropout2(self.feedforward2(self.act(self.dropout1(self.feedforward1(inputs)))))
            outputs += inputs  # 注意此处有残差机制
        return outputs
