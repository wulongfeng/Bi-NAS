import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class NAR(nn.Module):
    def __init__(self, args, user_num, item_num, factor_num, w2v_feat, item_feat_dim):
        super(NAR, self).__init__()
        self.args = args
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        self.q_user_proj = nn.Linear(factor_num, factor_num, bias=False)
        self.q_item_proj = nn.Linear(factor_num, factor_num, bias=False)

        self.atten_fusion = nn.Linear(item_feat_dim*2, item_feat_dim, bias=False)

        self.word_feat = torch.FloatTensor(w2v_feat)
        self.word_feat = self.word_feat.to(args.device)
        output_hidden_dim = factor_num+w2v_feat.shape[1]
        if args.single_output_layer:
            self.output_linear = nn.Linear(output_hidden_dim, 1)
        else:
            self.output_linear1 = nn.Linear(output_hidden_dim, output_hidden_dim//2)
            self.output_linear2 = nn.Linear(output_hidden_dim//2, 1)
        self.drop = nn.Dropout(p=args.dropout)
        self._init_weight_()
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)


    def _init_weight_(self):
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        self.embed_user.weight.requires_grad = True
        self.embed_item.weight.requires_grad = True

    def scaled_dot_product(self, input_embed, mask=None):
        q, k, v = self.q_proj(input_embed), self.k_proj(input_embed), self.v_proj(input_embed)

        d_k = q.size()[0]
        attn_logits = torch.matmul(q.transpose(-2, -1), k)
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(v, attention)

        return values, attention

    def forward(self, user, user_feat, item, item_feat):
        if self.args.norm_feat:
            user_feat = F.normalize(user_feat, p=2, dim=1)
            item_feat = F.normalize(item_feat, p=2, dim=1)

        user_embedding = self.embed_user(user)
        item_embedding = self.embed_item(item)
        user_embedding = self.q_user_proj(user_embedding) # 20, 300
        item_embedding = self.q_item_proj(item_embedding)

        # user_word_feat_att = F.softmax(torch.mm(user_embedding, self.word_feat.t()), dim=-1) # self.word_feat: 131,300
        user_word_feat_att = torch.mm(user_embedding, self.word_feat.t())
        item_word_feat_att = torch.mm(item_embedding, self.word_feat.t())
        user_fused_att = F.relu(self.atten_fusion(torch.cat([user_word_feat_att, user_feat], dim=-1)))
        item_fused_att = F.relu(self.atten_fusion(torch.cat([item_word_feat_att, item_feat], dim=-1)))

        user_word_embedding = torch.matmul(user_fused_att,self.word_feat)
        item_word_embedding = torch.matmul(item_fused_att,self.word_feat)

        # torch.einsum('ijk,ik->ij', z, b)
        if self.args.use_output_layer:
            feat_embed = user_word_embedding * item_word_embedding
            id_embed = user_embedding * item_embedding
            output_hidden = torch.cat([feat_embed, id_embed], dim=-1)
            output_hidden = self.drop(output_hidden)
            if self.args.single_output_layer :
                output = self.output_linear(output_hidden)
            else:
                output = self.output_linear2(F.relu(self.output_linear1(output_hidden)))
        else:
            user_factor = torch.cat([user_embedding, user_word_embedding], dim=1)
            item_factor = torch.cat([item_embedding, item_word_embedding], dim=1)
            user_factor = self.drop(user_factor)
            item_factor = self.drop(item_factor)
            output = (user_factor * item_factor).sum(dim=1)
        return output

    def vis(self, data):
        train_users = data.train_user_all
        user_feat = data.user_feature_all[train_users]
        if self.args.norm_feat:
            user_feat = F.normalize(user_feat, p=2, dim=1)

        train_users = torch.tensor(train_users).to(self.args.device)
        user_embedding = self.embed_user(train_users)
        user_embedding = self.q_user_proj(user_embedding)
        # user_word_feat_att = F.softmax(torch.mm(user_embedding, self.word_feat.t()), dim=-1).detach().cpu().numpy()
        user_word_feat_att = torch.mm(user_embedding, self.word_feat.t())
        user_fused_att = F.relu(self.atten_fusion(torch.cat([user_word_feat_att, user_feat], dim=-1))).detach().cpu().numpy()

        user_att_dict = defaultdict(list)
        for idx, user in enumerate(train_users):
            user_att_dict[user.item()] = user_fused_att[idx]

        return user_att_dict

    def compute_loss(self, user_batch, user_feature_batch, pos_item_batch, pos_item_feature_batch, neg_item_batch, neg_item_feature_batch, epoch=None):
        # compute positive socre
        pos_score = self.forward(user_batch, user_feature_batch, pos_item_batch, pos_item_feature_batch)
        # compute negative socre
        neg_score = self.forward(user_batch, user_feature_batch, neg_item_batch, neg_item_feature_batch)
        # compute loss
        loss = - torch.nn.functional.logsigmoid(pos_score - neg_score).mean()
        #print ("Logits: ", (pos_score - neg_score)[0])
        return loss