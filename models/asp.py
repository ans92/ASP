import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from sklearn.preprocessing import scale
import random
from .common import MLP
from .word_embedding import load_word_embeddings
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#np.random.seed(400)
#random.seed(400)
#torch.manual_seed(400)
#if torch.cuda.is_available():
#s    torch.cuda.manual_seed(400)

class ASP(nn.Module):
    def __init__(self, dset, args):
        super(ASP, self).__init__()
        self.obj_head = MLP(dset.feat_dim,1024,2,relu=True,dropout=True,norm=True,layers=[768,1024])
        self.attr_head = MLP(dset.feat_dim,1024,2,relu=True,dropout=True,norm=True,layers=[768,1024])
#        self.attr_clf = MLP(1024, len(dset.attrs), 1, relu = False)
#        self.obj_clf = MLP(1024, len(dset.objs), 1, relu = False)
        self.dset = dset
        self.args = args
        if dset.open_world:
            self.known_pairs = dset.train_pairs
            seen_pair_set = set(self.known_pairs)
            mask = [1 if pair in seen_pair_set else 0 for pair in dset.pairs]
        #     self.seen_mask = torch.BoolTensor(mask).cuda() * 1.
            self.seen_mask = torch.BoolTensor(mask).to(device) * 1.

        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(device), \
                                          torch.arange(len(self.dset.objs)).long().to(device)
        self.feasibility_scores = self.feasibility_embeddings()

        def get_attr_obj_ids(att, obj):
            objs = [dset.obj2idx[o] for o in obj]
            attrs = [dset.attr2idx[a] for a in att]
            attrs = torch.LongTensor(attrs).to(device)
            objs = torch.LongTensor(objs).to(device)
            
            return attrs, objs
        
        
        self.attrs, self.objs = get_attr_obj_ids(self.dset.attrs, self.dset.objs)
        
#        try:
#            self.args.fc_emb = self.args.fc_emb.split(',')
#        except:
#            self.args.fc_emb = [self.args.fc_emb]
#        layers = []
#        for a in self.args.fc_emb:
#            a = int(a)
#            layers.append(a)
#            
#        self.image_embedder = MLP(dset.feat_dim, int(args.emb_dim), relu=args.relu, num_layers=args.nlayers,
#                                  dropout=self.args.dropout,
#                                  norm=self.args.norm, layers=layers)
                                  
        input_dim = args.emb_dim
        self.attr_embedder = nn.Embedding(len(dset.attrs), input_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), input_dim)
        
        if args.emb_init:
            pretrained_weight = load_word_embeddings(args.emb_init, dset.attrs)
            self.attr_embedder.weight.data.copy_(pretrained_weight)
            pretrained_weight = load_word_embeddings(args.emb_init, dset.objs)
            self.obj_embedder.weight.data.copy_(pretrained_weight)
            
        self.norm1 = nn.LayerNorm(args.emb_dim)
        self.atten1 = nn.MultiheadAttention(args.emb_dim, num_heads=4)
#        self.atten2 = nn.MultiheadAttention(args.emb_dim, num_heads=4)
        
        self.attr_comp_proj = MLP(args.emb_dim,1024,2,relu=True,dropout=True,norm=True,layers=[768,1024]) #nn.Linear(args.emb_dim, 1024)
        self.obj_comp_proj = MLP(args.emb_dim,1024,2,relu=True,dropout=True,norm=True,layers=[768,1024]) #nn.Linear(args.emb_dim, 1024)
        
#        self.attr_img_proj = nn.Linear(args.emb_dim, args.emb_dim)
#        self.obj_img_proj = nn.Linear(args.emb_dim, args.emb_dim)
        
        self.linear1 = nn.Linear(args.emb_dim, args.emb_dim)
#        self.linear2 = nn.Linear(1024, 1024)
#        self.linear3 = nn.Linear(args.emb_dim, 1024)
#       self.linear4 = nn.Linear(4096, args.emb_dim)
        
        self.norm2 = nn.LayerNorm(args.emb_dim)
        self.norm3 = nn.LayerNorm(args.emb_dim)
#        self.norm4 = nn.LayerNorm(1024)
#        self.norm5 = nn.LayerNorm(args.emb_dim)
        
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
#        self.dropout3 = nn.Dropout(0.5)
#        self.dropout4 = nn.Dropout(0.5)
        
        self.activation = nn.ReLU(inplace=True)
        
    
    def compose(self, attrs, objs):
        
        attrs, objs = self.attr_embedder(attrs), self.obj_embedder(objs)
        
        return attrs, objs
    
    
    
    def train_forward(self, x):
        img, attrs, objs, mask = x[0],x[1], x[2],x[4]

        # attr_feats = self.attr_head(img)
        # obj_feats = self.obj_head(img)

        # attr_pred = self.attr_clf(attr_feats)
        # obj_pred = self.obj_clf(obj_feats)

        # if self.args.partial == True:
            # attr_loss = F.cross_entropy(attr_pred[mask==0,:], attrs[mask==0])
            # obj_loss = F.cross_entropy(obj_pred[mask==1,:], objs[mask==1])
        # else:
            # attr_loss = F.cross_entropy(attr_pred,attrs)
            # obj_loss = F.cross_entropy(obj_pred,objs)


        # if self.args.gumbel == True and self.args.partial == True:
            
            # obj_inf = 1.0*(torch.Tensor(self.feasibility_scores[:,attrs.cpu()]).cuda().permute(1,0))
            # obj_labels = F.gumbel_softmax(obj_inf* obj_pred, dim=-1, hard=True).argmax(-1).detach()
            # obj_weak_loss = F.cross_entropy(obj_pred[mask==0],obj_labels[mask==0])

            # att_inf = 1.0*(torch.Tensor(self.feasibility_scores[objs.cpu(),:]).cuda())
            # att_labels = F.gumbel_softmax(att_inf* attr_pred, dim=-1, hard=True).argmax(-1).detach()
            # att_weak_loss = F.cross_entropy(attr_pred[mask==1], att_labels[mask==1])
            # weak_loss = att_weak_loss + obj_weak_loss


        # loss =  attr_loss + obj_loss
        # if self.args.gumbel==True and self.args.partial == True:
            # loss = loss+weak_loss
        
#        img_feats = self.image_embedder(img)
        attr_img_feats = self.attr_head(img)
        obj_img_feats = self.obj_head(img)
        
        attr_feats_normed = F.normalize(attr_img_feats, dim=1)
#        attr_feats_normed = torch.unsqueeze(attr_feats_normed, 1)
        
        obj_feats_normed = F.normalize(obj_img_feats, dim=1)        
#        obj_feats_normed = torch.unsqueeze(obj_feats_normed, 1)
        
        att_emb, obj_emb = self.compose(self.attrs, self.objs)
        pair_embed = torch.cat((att_emb, obj_emb), 0)
        
        pair_embed = self.norm1(pair_embed)
#        pair_embed = torch.unsqueeze(pair_embed, 0)
#        pair_embed = pair_embed.repeat(img_feats_normed.shape[0], 1, 1)
#        
#        pair_embed = torch.cat((img_feats_normed, pair_embed), 1)
        
        attention, wgt = self.atten1(pair_embed, pair_embed, pair_embed)
        attention = pair_embed + self.dropout1(attention)
        
        attention = self.norm2(attention)
        
#        attention_ , wgt = self.atten2(attention, attention, attention)
#        attention = attention + self.dropout2(attention_)
       
        attention_ = self.dropout1(self.activation(self.linear1(attention)))
        attention = attention + attention_
        attention = self.norm3(attention)
#        attention = self.dropout2(self.activation(self.norm4(self.linear2(attention))))
#        attention = self.activation(self.linear3(attention))
        
#        attr_atten = attention[:,1:len(self.dset.attrs)+1,:]
#        obj_atten = attention[:,-len(self.dset.objs):,:]
        
        attr_atten = attention[:len(self.dset.attrs),:]
        obj_atten = attention[-len(self.dset.objs):,:]
        
#        img_feats = attention[:,:1,:]
#        attr_img_feats = self.attr_img_proj(img_feats_normed)
#        obj_img_feats = self.obj_img_proj(img_feats_normed)
        
        attr_atten = self.attr_comp_proj(attr_atten)
        obj_atten = self.obj_comp_proj(obj_atten)
        
#        attr_attention = torch.matmul(attr_atten, attr_img_feats.permute(0,2,1))
#        obj_attention = torch.matmul(obj_atten, obj_img_feats.permute(0,2,1))
#        
#        attr_attention = attr_attention.view(attr_attention.shape[0], attr_attention.shape[1])
#        obj_attention = obj_attention.view(obj_attention.shape[0], obj_attention.shape[1])
        
        attr_attention = torch.matmul(attr_img_feats, attr_atten.permute(1,0))
        obj_attention = torch.matmul(obj_img_feats, obj_atten.permute(1,0))
        
        attr_loss = F.cross_entropy(attr_attention, attrs)
        obj_loss = F.cross_entropy(obj_attention, objs)
        
        
        
        loss = attr_loss + obj_loss
        return loss, None

    def compute_feasibility(self):
        scores=np.load(self.args.kbfile,allow_pickle=True).item()
        feasibility_scores=[0 for i in range(len(self.dset.attrs)*len(self.dset.objs))]
        for a in self.dset.attrs:
            for o in self.dset.objs:
                score = scores[o][a]
                idx = self.dset.all_pair2idx[(a, o)]
                feasibility_scores[idx]=score

        self.feas_scores = feasibility_scores
        return feasibility_scores


    def feasibility_embeddings(self):

        scores = np.load(self.args.kbfile,allow_pickle=True).item()
        feasibility_scores = [[0 for i in range(len(self.dset.attrs))] for j in range(len(self.dset.objs))]
        for i in range(len(self.dset.objs)):
            for j in range(len(self.dset.attrs)):
                feasibility_scores[i][j]=max(scores[self.dset.objs[i]][self.dset.attrs[j]],0)
  
        return np.array(feasibility_scores)


    def val_forward_with_threshold(self, x, th=0.):
        img = x[0]
#        attr_pred = F.softmax(self.attr_clf(self.attr_head(img)), dim=1)
#        obj_pred = F.softmax(self.obj_clf(self.obj_head(img)), dim=1)
#
#        score = torch.bmm(attr_pred.unsqueeze(2), obj_pred.unsqueeze(1)).view(attr_pred.shape[0],-1)
        # Note: Pairs are already aligned here
        
        attr_img_feats = self.attr_head(img)
        obj_img_feats = self.obj_head(img)
        
#        img_feats_normed = F.normalize(img_feats, dim=1)
#        img_feats_normed = torch.unsqueeze(img_feats_normed, 1)        
        
        attr_feats_normed = F.normalize(attr_img_feats, dim=1)
        obj_feats_normed = F.normalize(obj_img_feats, dim=1)
        
        att_emb, obj_emb = self.compose(self.attrs, self.objs)
        pair_embed = torch.cat((att_emb, obj_emb), 0)
        
        pair_embed = self.norm1(pair_embed)
#        pair_embed = torch.unsqueeze(pair_embed, 0)
#        pair_embed = pair_embed.repeat(img_feats_normed.shape[0], 1, 1)
#        
#        pair_embed = torch.cat((img_feats_normed, pair_embed), 1)
        
        attention, wgt = self.atten1(pair_embed, pair_embed, pair_embed)
        attention = pair_embed + self.dropout1(attention)
        
        attention = self.norm2(attention)

        attention_ = self.activation(self.linear1(attention))
        attention = attention + attention_
        attention = self.norm3(attention)
#       
#        attention = self.dropout1(self.activation(self.norm3(self.linear1(attention))))
#        attention = self.dropout2(self.activation(self.norm4(self.linear2(attention))))
#        attention = self.activation(self.linear3(attention))
#        
#        attr_atten = attention[:,1:len(self.dset.attrs)+1,:]
#        obj_atten = attention[:,-len(self.dset.objs):,:]

        attr_atten = attention[:len(self.dset.attrs),:]
        obj_atten = attention[-len(self.dset.objs):,:]  
        
#        img_feats = attention[:,:1,:]
#        attr_img_feats = self.attr_img_proj(img_feats_normed)
#        obj_img_feats = self.obj_img_proj(img_feats_normed)
        
        attr_atten = self.attr_comp_proj(attr_atten)
        obj_atten = self.obj_comp_proj(obj_atten)
        
#        attr_attention = torch.matmul(attr_atten, attr_img_feats.permute(0,2,1))
#        obj_attention = torch.matmul(obj_atten, obj_img_feats.permute(0,2,1))
#        
#        attr_attention = attr_attention.view(attr_attention.shape[0], attr_attention.shape[1])
#        obj_attention = obj_attention.view(obj_attention.shape[0], obj_attention.shape[1])
        
        attr_attention = torch.matmul(attr_img_feats, attr_atten.permute(1,0))
        obj_attention = torch.matmul(obj_img_feats, obj_atten.permute(1,0))
        
        
        score = torch.bmm(attr_attention.unsqueeze(2), obj_attention.unsqueeze(1)).view(attr_attention.shape[0],-1)
        
        
        
        mask = torch.Tensor((np.array(self.feas_scores)>=th)*1.0).to(device)
        score = score*mask + (1.-mask)*(-1.)
        scores = {}
        for itr, (attr, obj) in enumerate(self.dset.pairs):
            attr_id, obj_id = self.dset.attr2idx[attr], self.dset.obj2idx[obj]
            idx = obj_id + attr_id * len(self.dset.objs)
            scores[(attr, obj)] = score[:, idx]
        return score, scores

    def val_forward_revised(self, x):
        img = x[0]
        # attr_pred = F.softmax(self.attr_clf(self.attr_head(img)), dim=1)
        # obj_pred = F.softmax(self.obj_clf(self.obj_head(img)), dim=1)
        # score = torch.bmm(attr_pred.unsqueeze(2), obj_pred.unsqueeze(1)).view(attr_pred.shape[0],-1)
        
#        img_feats = self.image_embedder(img)
        attr_img_feats = self.attr_head(img)
        obj_img_feats = self.obj_head(img)
        
#        img_feats_normed = F.normalize(img_feats, dim=1)
#        img_feats_normed = torch.unsqueeze(img_feats_normed, 1)        
        
        attr_feats_normed = F.normalize(attr_img_feats, dim=1)
        obj_feats_normed = F.normalize(obj_img_feats, dim=1)
        
        att_emb, obj_emb = self.compose(self.attrs, self.objs)
        pair_embed = torch.cat((att_emb, obj_emb), 0)
        
        pair_embed = self.norm1(pair_embed)
#        pair_embed = torch.unsqueeze(pair_embed, 0)
#        pair_embed = pair_embed.repeat(img_feats_normed.shape[0], 1, 1)
#        
#        pair_embed = torch.cat((img_feats_normed, pair_embed), 1)
        
        attention, wgt = self.atten1(pair_embed, pair_embed, pair_embed)
        attention = pair_embed + attention
        
        attention = self.norm2(attention)
        
#        attention_ , wgt = self.atten2(attention, attention, attention)
#        attention = attention + attention_
        
        attention_ = self.activation(self.linear1(attention))
        attention = attention + attention_
        attention = self.norm3(attention)
        
#        attention = self.dropout1(self.activation(self.norm3(self.linear1(attention))))
#        attention = self.dropout2(self.activation(self.norm4(self.linear2(attention))))
#        attention = self.activation(self.linear3(attention))
#        
#        attr_atten = attention[:,1:len(self.dset.attrs)+1,:]
#        obj_atten = attention[:,-len(self.dset.objs):,:]

        attr_atten = attention[:len(self.dset.attrs),:]
        obj_atten = attention[-len(self.dset.objs):,:]  
        
#        img_feats = attention[:,:1,:]
#        attr_img_feats = self.attr_img_proj(img_feats_normed)
#        obj_img_feats = self.obj_img_proj(img_feats_normed)
        
        attr_atten = self.attr_comp_proj(attr_atten)
        obj_atten = self.obj_comp_proj(obj_atten)
        
#        attr_attention = torch.matmul(attr_atten, attr_img_feats.permute(0,2,1))
#        obj_attention = torch.matmul(obj_atten, obj_img_feats.permute(0,2,1))
#        
#        attr_attention = attr_attention.view(attr_attention.shape[0], attr_attention.shape[1])
#        obj_attention = obj_attention.view(obj_attention.shape[0], obj_attention.shape[1])
        
        attr_attention = torch.matmul(attr_img_feats, attr_atten.permute(1,0))
        obj_attention = torch.matmul(obj_img_feats, obj_atten.permute(1,0))
        
        
        score = torch.bmm(attr_attention.unsqueeze(2), obj_attention.unsqueeze(1)).view(attr_attention.shape[0],-1)

        

        scores = {}
        for itr, (attr, obj) in enumerate(self.dset.pairs):
            attr_id, obj_id = self.dset.attr2idx[attr], self.dset.obj2idx[obj]
            idx = obj_id + attr_id * len(self.dset.objs)
            scores[(attr, obj)] = score[:, idx]
        return score, scores

    def forward(self, x,threshold=None):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                if threshold is None:
                    loss, pred = self.val_forward_revised(x)
                else:
                    loss, pred = self.val_forward_with_threshold(x,threshold)

        return loss, pred
