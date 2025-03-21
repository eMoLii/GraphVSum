#!/usr/bin/env python
"""
decoder.py: define Decoder class that will be used to encode
the input sequence.
Note: Both video and text modality decoder are available.
"""

import numpy as np
import torch
import warnings
import torch.nn as nn
import dgl.nn.pytorch as dglnn
from typing import List, Optional
from torch.nn import functional as F
from models.graphvsum.positional_encoding import PositionalEncoding
from models.graphvsum.custom_encoder import CustomTransformerEncoder, _get_activation
from models.graphvsum.custom_transformer import TransformerEncoderLayer
from dataloader.multimodal_dataset import load_graph


class RGATModel(nn.Module):
    def __init__(self, d_model, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(d_model, d_model * 2, 2)
            for rel in rel_names}, aggregate='sum')
        # self.conv3 = dglnn.HeteroGraphConv({
        #     rel: dglnn.GATConv(d_model // 4, d_model // 4, 2)
        #     for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(d_model * 2, d_model, 2)
            for rel in rel_names}, aggregate='sum')
        # self.dropout = nn.Dropout(p = 0.2)

    def forward(self, graph, inputs):
        # 输入是节点的特征字典
        h = self.conv1(graph, inputs)
        h = {k: v.mean(1) for k, v in h.items()}
        h = {k: F.relu(v) for k, v in h.items()}
        h = {k: F.dropout(v, p=0.1, training=self.training) for k, v in h.items()}
        # h = self.conv3(graph, h)
        # h = {k: v.mean(1) for k, v in h.items()}
        # h = {k: F.relu(v) for k, v in h.items()}
        h = {k: F.dropout(v, p=0.1, training=self.training) for k, v in h.items()}  
        h = self.conv2(graph, h)
        h = {k: v.mean(1) for k, v in h.items()}
        return h

# class AttentionFusion(nn.Module):  
#     def __init__(
#             self,
#     ):  
#         super(AttentionFusion, self).__init__()  
#         # 定义可学习的参数 alpha  
#         self.attentiond = nn.Sequential(
#             nn.Linear(128 * 2, 128),  # 将X1和X2拼接后通过线性变换
#             nn.ReLU(),
#             nn.Linear(128, 2),  # 输出两个权重，用softmax进行归一化
#             nn.Softmax(dim=-1)  # 在最后一个维度进行归一化
#         )
#         self.attentionv = nn.Sequential(
#             nn.Linear(128 * 2, 128),  # 将X1和X2拼接后通过线性变换
#             nn.ReLU(),
#             nn.Linear(128, 2),  # 输出两个权重，用softmax进行归一化
#             nn.Softmax(dim=-1)  # 在最后一个维度进行归一化
#         )
#         # self.rulu = nn.ReLU()
#         # self.sigmoid = nn.Sigmoid()
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.trunc_normal_(m.weight, std=.05)

#     def forward(self, batch_tfd_feats, batch_tfv_feats, batch_gd_feats, batch_gv_feats):
#         fusion_inputd = torch.cat((batch_tfd_feats, batch_gd_feats), dim=-1)
        
#         # 计算注意力权重
#         attention_weightsd = self.attentiond(fusion_inputd)  # [batch_size, 2]
        
#         # 对 X1 和 X2 应用注意力权重
#         tfd_weighted = attention_weightsd[:, :, 0].unsqueeze(-1) * batch_tfd_feats
#         gd_weighted = attention_weightsd[:, :, 1].unsqueeze(-1) * batch_gd_feats
        
#         # 融合特征
#         hd = tfd_weighted + gd_weighted
        
#         fusion_inputv = torch.cat((batch_tfv_feats, batch_gv_feats), dim=-1)
        
#         # 计算注意力权重
#         attention_weightsv = self.attentionv(fusion_inputv)  # [batch_size, 2]
        
#         # 对 X1 和 X2 应用注意力权重
#         tfv_weighted = attention_weightsv[:, :, 0].unsqueeze(-1) * batch_tfv_feats
#         gv_weighted = attention_weightsv[:, :, 1].unsqueeze(-1) * batch_gv_feats
        
#         # 融合特征
#         hv = tfv_weighted + gv_weighted

#         return hd, hv



class FFN(nn.Module):
    def __init__(
        self,
        in_features = 128,
        out_features = 1,
    ):
        super().__init__()
        self.linear1 = nn.Linear(in_features, in_features, bias=True).float()
        self.linear2 = nn.Linear(in_features, out_features, bias=True).float()
        self.rulu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.05)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.05)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.linear1(x)
        h = self.dropout(h)
        h = self.rulu(h)
        y = self.linear2(h)
        return y


class decoder(nn.Module):
    r"""
    Takes shot embeg and dialogue embedding from `(n-1)th` EPISODE and
    try to classify baddinsed on whether shots should be taken into RECAP or not.
    This is an all-purpose decoder which can be used for only `video` or `dialogue`
    modalities or `both`.
    ----------------------------------------------------------------------------
    `Note:` that Transformer's `encoder` and `decoder` differs only at the
    `cross-attention` technique.
    """

    def __init__(self,
                 vid_feat_dim: int,
                 dia_feat_dim: int,
                 d_model: int = 128,
                 num_heads: int = 8,
                 ffn_ratio: float = 4.0,
                 modality: str = 'both',
                 withGROUP: bool = True,
                 attention_type: str = 'full',
                 differential_attention: bool = False,
                 differential_attention_type: str = 'basic',
                 max_groups: int = 100,
                 max_pos_enc_len: int = 4000,
                 num_layers: int = 6,
                 drop_vid: float = 0.1,
                 drop_dia: float = 0.1,
                 drop_trm: float = 0.2,
                 drop_fc: float = 0.4,
                 activation_trm: str = "gelu",
                 activation_mlp: str = "gelu",
                 activation_clf: str = "relu",
                 hidden_sizes: List = [],
                 ) -> None:
        r"""
        ----------------------------------------------------------------------------
        Args:
            - vid_feat_dim: Dimension of video feature vector (e.g., 512).
            - dia_feat_dim: Dimension of dialogue feature vector (e.g., 512).
            - d_model: Dimension of the common feature space. `default=128`.
            - num_heads: number of heads in Transformer layers. `default=8`.
            - ffn_ratio: Ratio of `dim_feedforward` to `d_model` in
                `TransformerEncoderLayer`. `default=4.0`.
            - modality: Type of modality to be used. Available options are `vid` for video,
              `dia` for dialogue, or both. `default='both'`.
            - withGROUP: Whether to add GROUP (or <SEP> informally) token or not. `default=True`
            - attention_type: Type of attention to be used. Available options are
              `full` and `sparse`. `default='full'`.
            - differential_attention: When `True`, each encoder layer will have different
              `src_attention` mask. It shhould be used when `attention_type = sparse`.
              `default=False`.
            - differential_attention_type: Type of differential attention to be used.
              `basic` repeats the src_attention mask for alternate layer, while in `advanced`
              the attention mask for 3rd is combination of 1st and 2nd. Should be given when `differential_attention = True`. `default='basic'`.
            - max_groups: Maximum number of groups to be considered. `default=100`.
            - num_layers: number of Transformer decoder layers.
            - max_pos_enc_len: maximum length of positional encoding.
            - drop_vid: dropout ratio while projecting video features. `default=0.1`.
            - drop_dia: dropout ratio while projecting dialogue features. `default=0.1`.
            - drop_trm: dropout ratio for Transformer layers. `default=0.2`.
            - drop_fc: dropout ratio for MLP layers. `default=0.4`.
            - activation_trm: activation function for Transformer layers. `default=gelu`.
            - activation_mlp: activation function for MLP layers (during common space projection).
              `default=gelu`.
            - activation_clf: activation function for classification head. `default=relu`.
            - hidden_sizes: MLP hidden-layer sizes in form of a list. This maps the output vector
              in `n-dim` from `decoder` to `one` dimensional. `default=[]`
        """
        super(decoder, self).__init__()
        self.d_model = d_model
        self.pos_decoder = PositionalEncoding(d_model, max_len=max_pos_enc_len)
        decoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=num_heads,
                                                dim_feedforward=ffn_ratio*d_model,
                                                dropout=drop_trm,
                                                activation=_get_activation(activation_trm),
                                                batch_first=True)

        if attention_type == "sparse" and differential_attention and num_layers%3 != 0:
            warnings.warn(message=('`num_layers` should be multiple of 3 when `differential_attention` is True and '
                                   '`attention_type` is sparse. Setting `differential_attention` to False.'),
                          category=UserWarning)
            differential_attention = False

        self.transformer_decoder = CustomTransformerEncoder(decoder_layer,
                                                            num_layers=num_layers,
                                                            norm=nn.LayerNorm(d_model),
                                                            per_layer_src_mask=differential_attention)

        # Type embeddings for video, dialogue, [GROUP], and <PAD> tokens.
        # {'vid': 0, 'dia': 1, 'cls': 2, 'pad': 3}
        self.emb = nn.Embedding(num_embeddings=4, embedding_dim=d_model, padding_idx=3)
        self.modality = modality
        self.withGROUP = withGROUP
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attention_type = attention_type
        self.differential_attention = differential_attention
        self.differential_attention_type = differential_attention_type
        # Group type embeddings
        # NOTE: If you want different initializations for each [GROUP] token
        # then you can use another `nn.Embedding` with `num_embeddings=max_groups`.
        # The fact is, `group_type` along with group embedding i.e., idx=2 of `self.emb` are sufficient.
        if self.attention_type == 'sparse':
            self.group_idx_emb = nn.Embedding(num_embeddings=max_groups,
                                              embedding_dim=d_model,
                                              padding_idx=0)

        # MLP - to project into d_model dim space.
        if vid_feat_dim != d_model and modality != 'dia':
            self.visMLP = nn.Sequential(nn.Linear(vid_feat_dim, d_model), _get_activation(activation_mlp), nn.Dropout(drop_vid))
        if dia_feat_dim != d_model and modality != 'vid':
            self.diaMLP = nn.Sequential(nn.Linear(dia_feat_dim, d_model), _get_activation(activation_mlp), nn.Dropout(drop_dia))

        # Calssification head MLP
        linear_layers = []
        old_size = d_model
        for size in hidden_sizes:
            linear_layers.extend([nn.Linear(old_size, size), _get_activation(activation_clf), nn.Dropout(drop_fc)])
            old_size = size
        linear_layers.append(nn.Linear(old_size, 1))
        self.mlp = nn.Sequential(*linear_layers)

        self.model_RGAT = RGATModel(d_model, ['dia_sim', 'time_align', 'time_align_by', 'video_sim'])
        # self.feture_fusion = AttentionFusion()
        self.d_FFN = FFN(d_model)
        self.v_FFN = FFN(d_model)
        self.res_weight = 0
        self.criterion = nn.BCEWithLogitsLoss()


    def forward(self,
                vid_feats: Optional[torch.Tensor],
                dia_feats: Optional[torch.Tensor],
                video_name,
                mask: torch.Tensor,
                time_idx: torch.Tensor,
                token_type_ids,
                group_ids: torch.Tensor,
                subseq_len: torch.Tensor,
                edge_dict,
                dia_targets,
                dia_boolean_mask,
                vid_targets,
                vid_boolean_mask) -> torch.Tensor:
        r"""
        Args:
            - vid_feats: of size `(b = batch of episodes, m = no. of shots, feature_size = e.g., 512)`
            - dia_feats: of size `(b, n, feature_size = e.g., 512)`. It should follow the
              temporal order of dialogue. Same goes for visual features.
            - mask: of size `(b, m+n+eps)`. `1` for relevant and `0` for irrelevant.
              Note: Here `eps` is a small number to account for the case for `GROUP` tokens.
            - time_idx: of size `(b, m+n+eps)`. Relative time index of each shot and dialogue for each episode.
            - token_type_ids: of size `(b, m+n+eps)`. `0` for video, `1` for dialogue, `2` for `[CLS]`.
            - group_ids: of size `(b, m+n+eps)`. a positive integer index for whole group/story-segment.
            - subseq_len: of size `(b, k, 2)`. `k` is the number of sub-sequences (or group). `2` is for
              length of `vid` and `dia` tokens for each sub-sequence.
        """

        device = mask.device
        batch_len, seq_len = mask.shape

        dia_num_count = torch.sum(dia_boolean_mask, dim=1)
        video_num_count = torch.sum(vid_boolean_mask, dim=1)
        max_d_num = torch.max(dia_num_count).item()
        max_v_num = torch.max(video_num_count).item()
        max_vd_num = torch.max(video_num_count + dia_num_count).item()

        hetero_graphs = []
        node_embs = []
        dv_time_align_list = []
        graph_densitys = []

        batch_feats = torch.zeros(0, max_vd_num, self.d_model).to(device)
        vd_times = torch.zeros(0, max_vd_num).to(device)
        my_token_type_ids = torch.zeros(0, max_vd_num).to(device)
        for i in range(batch_len):
            # if(video_name[i] == "prison-break_S02_S02E19"): # 
            #     x = 1

            dia_feat = dia_feats[i][dia_boolean_mask[i]].unsqueeze(0)
            vid_feat = vid_feats[i][vid_boolean_mask[i]].unsqueeze(0)
            
            vd_feat = torch.cat([vid_feat, dia_feat], dim = 1)
            vd_feat = torch.cat([vd_feat, torch.zeros(1, max_vd_num - vd_feat.shape[1], self.d_model).to(device)], dim = 1)
            batch_feats = torch.cat([batch_feats, vd_feat], dim = 0)
            
            video_time = time_idx[i][token_type_ids[i] == torch.tensor(0)].squeeze(0)
            dia_time = time_idx[i][token_type_ids[i] == torch.tensor(1)].squeeze(0)
            vd_time = torch.cat([video_time, dia_time])
            vd_time = torch.cat([vd_time, torch.zeros(max_vd_num - vd_time.shape[0]).to(device)]).unsqueeze(0)
            vd_times = torch.cat([vd_times, vd_time], dim = 0)

            token_type = torch.cat([torch.zeros(1, video_num_count[i].item()), torch.ones(1, dia_num_count[i].item())], dim = 1).to(device)
            token_type1 = torch.cat([token_type, 3 * torch.ones(1, max_vd_num - token_type.shape[1]).to(device)], dim = 1)
            my_token_type_ids = torch.cat([my_token_type_ids, token_type1], dim = 0)


            graph_label = {}
            graph_label['dia'] = dia_targets[i][dia_boolean_mask[i]]
            graph_label['video'] = vid_targets[i][vid_boolean_mask[i]]
            hetero_graph, dv_time_align, density = load_graph(video_name[i], edge_dict, graph_label)
            hetero_graphs.append(hetero_graph)
            graph_densitys.append(density)
            dv_time_align_list.append(dv_time_align.to(device))

            # node_embs.append({'dia': dia_feat, 'video':vid_feat})


        # batch_tfd_feats = torch.zeros(0, max_d_num, self.d_model).to(device)
        # batch_tfv_feats = torch.zeros(0, max_v_num, self.d_model).to(device)
        # for i in range(batch_len):
        #     v_feat = batch_feats[i][:video_num_count[i]]
        #     v_feat = torch.cat([v_feat, torch.zeros(max_v_num - v_feat.shape[0], self.d_model).to(device)], dim = 0)
        #     batch_tfv_feats = torch.cat([batch_tfv_feats, v_feat.unsqueeze(0)], dim = 0)

        #     d_feat = batch_feats[i][video_num_count[i]:video_num_count[i] + dia_num_count[i]]
        #     d_feat = torch.cat([d_feat, torch.zeros(max_d_num - d_feat.shape[0], self.d_model).to(device)], dim = 0)
        #     batch_tfd_feats = torch.cat([batch_tfd_feats, d_feat.unsqueeze(0)], dim = 0)            



        # hd =  batch_tfd_feats
        # hv =  batch_tfv_feats
        # d_out = self.d_FFN(hd).squeeze(dim=-1)
        # v_out = self.v_FFN(hv).squeeze(dim=-1)

        # return v_out, d_out, hd, hv, hetero_graphs




        batch_feats = self.pos_decoder(batch_feats, idx_to_choose=vd_times)
        batch_feats += self.emb(my_token_type_ids.to(torch.long))

        src_attn_mask = torch.zeros((0, max_vd_num, max_vd_num)).to(device)
        for i in range(batch_len):
            sample_mask = torch.zeros((max_vd_num, max_vd_num)).to(device)
            time_align_mask = torch.zeros((max_vd_num, max_vd_num), dtype=torch.bool).to(device)
            video_num = video_num_count[i].item()
            dia_num = dia_num_count[i].item()
            sample_mask[0:video_num, 0:video_num] = 1
            sample_mask[video_num:video_num+dia_num, video_num:video_num+dia_num] = 1
            
            time_align_mask[video_num:video_num+dia_num, 0:video_num] = dv_time_align_list[i]
            time_align_mask[0:video_num, video_num:video_num+dia_num] = dv_time_align_list[i].transpose(0, 1)
            sample_mask = torch.logical_or(sample_mask, time_align_mask)
            
            src_attn_mask = torch.cat([src_attn_mask, sample_mask.unsqueeze(0)], dim=0)

        src_attn_mask = torch.repeat_interleave(src_attn_mask, self.num_heads, dim=0)
        tf_out = self.transformer_decoder(batch_feats,
                                          mask=src_attn_mask.logical_not())
    

        batch_tfd_feats = torch.zeros(0, max_d_num, self.d_model).to(device)
        batch_tfv_feats = torch.zeros(0, max_v_num, self.d_model).to(device)
        for i in range(batch_len):
            v_feat = tf_out[i][:video_num_count[i]]
            v_feat = torch.cat([v_feat, torch.zeros(max_v_num - v_feat.shape[0], self.d_model).to(device)], dim = 0)
            batch_tfv_feats = torch.cat([batch_tfv_feats, v_feat.unsqueeze(0)], dim = 0)

            d_feat = tf_out[i][video_num_count[i]:video_num_count[i] + dia_num_count[i]]
            d_feat = torch.cat([d_feat, torch.zeros(max_d_num - d_feat.shape[0], self.d_model).to(device)], dim = 0)
            batch_tfd_feats = torch.cat([batch_tfd_feats, d_feat.unsqueeze(0)], dim = 0)            



        for i in range(batch_len):
            dia_feat = dia_feats[i][dia_boolean_mask[i]].unsqueeze(0)
            vid_feat = vid_feats[i][vid_boolean_mask[i]].unsqueeze(0)
            graph_dia_feat = batch_tfd_feats[i:i+1][:,:dia_feat.shape[1],:]
            graph_vid_feat = batch_tfv_feats[i:i+1][:,:vid_feat.shape[1],:]
            node_embs.append({'dia': graph_dia_feat, 'video':graph_vid_feat})



        batch_gd_feats = torch.zeros(0, max_d_num, self.d_model).to(device)
        batch_gv_feats = torch.zeros(0, max_v_num, self.d_model).to(device)

        for i in range(batch_len):
            graph_out = self.model_RGAT(hetero_graphs[i], node_embs[i])
            [dia_graph_out, video_graph_out] = [graph_out['dia'], graph_out['video']]
            d_feat = torch.cat([dia_graph_out, torch.zeros(max_d_num - dia_graph_out.shape[0], self.d_model).to(device)], dim = 0).unsqueeze(0)
            batch_gd_feats = torch.cat([batch_gd_feats, d_feat], dim = 0)            
            v_feat = torch.cat([video_graph_out, torch.zeros(max_v_num - video_graph_out.shape[0], self.d_model).to(device)], dim = 0).unsqueeze(0)
            batch_gv_feats = torch.cat([batch_gv_feats, v_feat], dim = 0)

        # hd, hv = self.feture_fusion(batch_tfd_feats, batch_tfv_feats, batch_gd_feats, batch_gv_feats)
        hd =  batch_gd_feats
        hv =  batch_gv_feats
        d_out = self.d_FFN(hd).squeeze(dim=-1)
        v_out = self.v_FFN(hv).squeeze(dim=-1)


        
        B, _ = v_out.shape
        loss = 0
        for i in range(B):
            loss += self.criterion(v_out[i][vid_boolean_mask[i]], vid_targets[i][vid_boolean_mask[i]])

        for i in range(B):
            loss += self.criterion(d_out[i][dia_boolean_mask[i]], dia_targets[i][dia_boolean_mask[i]])


        return v_out, d_out, hd, hv, hetero_graphs
    