''' Define the Transformer model '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import Constants
import pickle
from Layers import EncoderLayer, DecoderLayer
from torch.autograd import Variable

cudaid=1
vocabulary = pickle.load(open('./vocab.dict', 'rb'))
doc_vec_dict = pickle.load(open('./doc2vec.dict', 'rb'))
docid_dict = pickle.load(open('./docid.dict', 'rb'))
docid_dict[0] = 0
doc_vec_size = len(doc_vec_dict)


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def kernel_mus(n_kernels):
        l_mu = [1]
        if n_kernels == 1:
            return l_mu
        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

def kernel_sigmas(n_kernels):
    bin_size = 2.0 / (n_kernels - 1)
    l_sigma = [0.001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma
    l_sigma += [0.1] * (n_kernels - 1)
    return l_sigma

class knrm(nn.Module):
    def __init__(self, k):
        super(knrm, self).__init__()
        tensor_mu = torch.FloatTensor(kernel_mus(k)).cuda(cudaid)
        tensor_sigma = torch.FloatTensor(kernel_sigmas(k)).cuda(cudaid)
        self.mu = Variable(tensor_mu, requires_grad = False).view(1, 1, 1, k)
        self.sigma = Variable(tensor_sigma, requires_grad = False).view(1, 1, 1, k)
        self.dense = nn.Linear(k, 1, 1)

    def get_intersect_matrix(self, q_embed, d_embed, attn_q, attn_d):
        sim = torch.bmm(q_embed, torch.transpose(d_embed, 1, 2)).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[1], 1) # n*m*d*1
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * attn_d
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * 0.01 * attn_q
        log_pooling_sum = torch.sum(log_pooling_sum, 1)#soft-TF
        return log_pooling_sum

    def forward(self, inputs_q, inputs_d, mask_q, mask_d):
        q_embed_norm = F.normalize(inputs_q, 2, 2)
        d_embed_norm = F.normalize(inputs_d, 2, 2)
        mask_d = mask_d.view(mask_d.size()[0], 1, mask_d.size()[1], 1)
        mask_q = mask_q.view(mask_q.size()[0], mask_q.size()[1], 1)
        log_pooling_sum = self.get_intersect_matrix(q_embed_norm, d_embed_norm, mask_q, mask_d)
        output = F.tanh(self.dense(log_pooling_sum))
        return output

class Encoder_high(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_emb, src_pos, return_attns=False, needpos=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_pos, seq_q=src_pos)
        non_pad_mask = get_non_pad_mask(src_pos)

        # -- Forward
        if needpos:
            enc_output = src_emb + self.position_enc(src_pos)
        else:
            enc_output = src_emb

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class fuse(nn.Module):
    def __init__(self, vec_len) :
        super().__init__()
        self.vec_len = vec_len
        self.gate=nn.Sequential(nn.Linear(self.vec_len*2, 1), nn.Sigmoid())
        nn.init.xavier_normal_(self.gate[0].weight)  
    def forward(self, vec_a, vec_b) :
        z = self.gate(torch.cat([vec_a, vec_b], 1))#batch
        vec_c = z * vec_a + (1 - z) * vec_b#batch, vec
        return vec_c


class diver(nn.Module):
    
    def __init__(self, max_qdnum_cur, vec_len, mult) :
        super().__init__()
        self.mult = mult
        self.vec_len = vec_len
        self.max_qdnum_cur = max_qdnum_cur
        self.W = nn.Linear(vec_len, vec_len * mult, bias = False)
        self.aggr = nn.Sequential(nn.Linear(max_qdnum_cur, 1), nn.Tanh())
        nn.init.xavier_normal_(self.W.weight)   
        nn.init.xavier_normal_(self.aggr[0].weight)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, current_denc, docs1, docs2) :
        mult_current_denc = self.W(current_denc).view(-1, self.max_qdnum_cur * self.mult, self.vec_len)
        
        mult_sim_matrix = torch.bmm(mult_current_denc, torch.transpose(current_denc, 1, 2))#batch, max_qdnum_cur * mult, max_qdnum_cur
        mult_sim_matrix = mult_sim_matrix.view(-1, self.max_qdnum_cur, self.mult, self.max_qdnum_cur)
        mult_sim_matrix = torch.transpose(mult_sim_matrix, 1, 2)#batch, mult, max_qdnum_cur, max_qdnum_cur
        mult_sim_matrix = self.softmax(mult_sim_matrix)

        docs_1_idx = docs1.repeat(self.max_qdnum_cur * self.mult, 1).transpose(0, 1).view(-1, self.mult, 1, self.max_qdnum_cur)#batch, mult, 1, maxqdnum_cur
        docs_2_idx = docs2.repeat(self.max_qdnum_cur * self.mult, 1).transpose(0, 1).view(-1, self.mult, 1, self.max_qdnum_cur)#batch, mult, 1, maxqdnum_cur
        
        mult_docs_1_sim = torch.gather(mult_sim_matrix, 2, docs_1_idx).squeeze(2)#batch, mult, qdnum_cur
        mult_docs_2_sim = torch.gather(mult_sim_matrix, 2, docs_2_idx).squeeze(2)#batch, mult, qdnum_cur
        
        docs_1_score = 1 - self.aggr(mult_docs_1_sim).squeeze(2)#batch mult
        docs_2_score = 1 - self.aggr(mult_docs_2_sim).squeeze(2)#batch mult
        
        return docs_1_score, docs_2_score
    

class Contextual(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    
    def load_rep_embedding(self): # load the docment doc2vec embedding
        weight = torch.zeros(doc_vec_size + 1, 300)
        weight[-1] = torch.rand(300)
        for key in doc_vec_dict :
            weight[docid_dict[key], :] = doc_vec_dict[key]
        print("Successfully load the doc vectors...")
        return weight

    def load_embedding(self): # load the pretrained embedding
        weight = torch.zeros(len(vocabulary)+1, self.d_word_vec)
        weight[-1] = torch.rand(self.d_word_vec)
        with open('./wordid.txt', 'r') as fr:
            for line in fr:
                line = line.strip().split()
                wordid = vocabulary[line[0]]
                weight[wordid, :] = torch.FloatTensor([float(t) for t in line[1:]]) 
        print("Successfully load the word vectors...")
        return weight

    def __init__(
            self, max_querylen, max_doclen, max_qdlen, max_hislen, max_sessionlen, max_qdnum, batch_size, max_ddlen_cur, max_qdnum_cur, feature_len, 
            d_word_vec, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):

        super().__init__()

        self.max_qdlen = max_qdlen
        self.max_ddlen = max_qdlen - max_querylen
        self.max_qdnum = max_qdnum
        self.max_querylen = max_querylen
        self.max_doclen = max_doclen
        self.max_hislen = max_hislen
        self.max_sessionlen = max_sessionlen
        self.d_word_vec = d_word_vec
        self.batch_size = batch_size
        self.max_ddlen_cur = max_ddlen_cur
        self.max_qdnum_cur = max_qdnum_cur

        self.knrm_1 = knrm(11)
        self.knrm_2 = knrm(11)
        

        self.encoder_term = Encoder_high(
            len_max_seq=max_ddlen_cur,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.encoder_document = Encoder_high(
            len_max_seq=self.max_qdnum_cur,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)
        
        self.encoder_qd = Encoder_high(
            len_max_seq=self.max_qdnum_cur + 1,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)
        
        self.encoder_short_his = Encoder_high(
            len_max_seq=max_sessionlen+1,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=1, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.encoder_long_his = Encoder_high(
            len_max_seq=max_hislen+1,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=1, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.feature_layer=nn.Sequential(nn.Linear(feature_len, 1), nn.Tanh())
        self.gate=nn.Sequential(nn.Linear(self.d_word_vec*2, 1), nn.Sigmoid())

        self.div_mult = 4
        self.score_layer_p=nn.Linear(7, 1)
        self.score_layer_d=nn.Linear(self.div_mult, 1)
        self.embedding = nn.Embedding(len(vocabulary)+1, self.d_word_vec)
        self.click_embedding = nn.Embedding(3, self.d_word_vec)
        self.embedding.weight.data.copy_(self.load_embedding())
        self.rep_embedding = nn.Embedding(doc_vec_size+1, 300)
        self.rep_embedding.weight.data.copy_(self.load_rep_embedding())
        self.rep_projection = nn.Linear(300, self.d_word_vec)
        
        nn.init.xavier_normal_(self.feature_layer[0].weight)   
        nn.init.xavier_normal_(self.rep_projection.weight)   
        nn.init.xavier_normal_(self.score_layer_p.weight)   
        nn.init.xavier_normal_(self.score_layer_d.weight)   
        
        self.fuse_layer = fuse(vec_len=self.d_word_vec)

        self.div_layer = diver(max_qdnum_cur = max_qdnum_cur, vec_len=self.d_word_vec, mult=self.div_mult)

        
        for param in self.embedding.parameters():
            param.requires_grad = False
            
        for param in self.rep_embedding.parameters():
            param.requires_grad = False

    def pairwise_loss(self, score1, score2):
        return (1/(1+torch.exp(score2-score1)))


    def forward(self, query, docs1, docs2, features1, features2, long_qdids, long_click, long_pos, long_doc_pos, long_doc_vec, short_qdids, short_click, short_pos, short_doc_pos, short_doc_vec, current_dids, current_pos, current_vec):
        '''
        print(query.size())#batch, max_querylen
        print(docs1.size())#batch 
        print(docs2.size())#batch
        print(features1.size())#batch, featurelen
        print(features2.size())#batch, featurelen
        print(long_qdids.size())#batch, max_hislen, max_qdlen
        print(long_click.size())#batch, max_hislen, self.max_qdnum
        print(long_pos.size())#batch, max_hislen + 1
        print(long_doc_pos.size())#batch, max_hislen, self.max_qdnum
        print(long_doc_vec.size())#batch, max_hislen, self,max_qdnum

        print(short_qdids.size())#batch, max_sessionlen + 1, max_qdlen
        print(short_click.size())#batch, max_sessionlen + 1, self.max_qdnum
        print(short_pos.size())#batch, max_sessionlen + 1
        print(short_doc_pos.size())#batch, max_sessionlen + 1, self.max_qdnum
        print(short_doc_vec.size())#batch, max_sessionlen, self,max_qdnum

        print(current_dids.size())#batch, max_ddlen_cur
        print(current_pos.size())#batch, self.max_qdnum_cur
        print(current_vec.size())#batch, self.max_qdnum_cur
        '''
        current_dids_init = current_dids.view(-1, self.max_qdnum_cur, self.max_doclen)
        
        docs_1_idx = docs1.repeat(self.max_doclen, 1).transpose(0, 1).view(-1, 1, self.max_doclen)#batch, 1, max_doclen
        docs_2_idx = docs2.repeat(self.max_doclen, 1).transpose(0, 1).view(-1, 1, self.max_doclen)#batch, 1, max_doclen
        docs1_init = torch.gather(current_dids_init, 1, docs_1_idx).squeeze(1)#batch, maxdoclen
        docs2_init = torch.gather(current_dids_init, 1, docs_2_idx).squeeze(1)#batch, maxdoclen
        
        all_qdids = torch.cat([long_qdids, short_qdids], 1)#batch, max_hislen + max_sessionlen, max_qdlen
        all_click = torch.cat([long_click, short_click], 1)#batch, max_hislen + max_sessionlen, self.max_qdnum
        all_doc_vec = torch.cat([long_doc_vec, short_doc_vec], 1).view(-1, self.max_qdnum)#batch * (max_hislen + max_sessionlen), self.max_qdnum
        all_doc_pos = torch.cat([long_doc_pos, short_doc_pos], 1).view(-1, self.max_qdnum)#batch * (max_hislen + max_sessionlen), self.max_qdnum
        all_qd_mask = all_qdids.view(-1, self.max_qdlen)
        current_d_mask = current_dids.view(-1, self.max_ddlen_cur)
          
        all_qdenc = self.embedding(all_qdids)#batch, max_hislen + max_sessionlen +1, max_qdlen, vec
        all_doc_vec_embedding = self.rep_projection(self.rep_embedding(all_doc_vec))#batch * (max_hislen + max_sessionlen), self.max_qdnum, vec
        
        all_current_denc = self.embedding(current_dids)#batch, max_ddlen_cur, vec
        current_doc_vec_embedding = self.rep_projection(self.rep_embedding(current_vec))#batch, max_qdnum_cur, vec
        
        current_q_i = self.embedding(query)#batch, max_querylen, vec
        
        all_click_embedding = self.click_embedding(all_click).view(-1, self.max_qdnum, self.d_word_vec)#(batch * maxhislen + max_session), self.max_qdnum, vec
        
        current_q_vec_i, *_ =  self.encoder_term(current_q_i, query)#batch, max_querylen, vec
        current_q_vec = torch.sum(current_q_vec_i, 1)#batch, vec

        all_qdenc = all_qdenc.view(-1, self.max_qdlen, self.d_word_vec)#batch * (max_hislen + max_sessionlen), max_qdlen, vec
        all_qdenc_term, *_ = self.encoder_term(all_qdenc, all_qd_mask)#barch * (maxhis + max_session), max_qdlen, vec
        all_qenc_term, all_denc_term = torch.split(all_qdenc_term, [self.max_querylen, self.max_ddlen], 1)
        
        all_denc_term = all_denc_term.contiguous().view(-1, self.max_qdnum, self.max_doclen, self.d_word_vec)#batch * (maxhis + max_session), self.max_qdnum, max_doclen, vec
        all_denc_term = torch.sum(all_denc_term, 2)#batch * (maxhis + maxsession), self.max_qdnum, vec
        all_qenc_term = torch.sum(all_qenc_term, 1)#barch * (maxhis + maxsession), vec
                
        all_denc_vec, *_ = self.encoder_document(all_denc_term + all_click_embedding + all_doc_vec_embedding, all_doc_pos, needpos=True)#batch * (maxhis + maxsession), self.max_qdnum, vec
        all_qdenc_vec = torch.cat([all_qenc_term.unsqueeze(1), all_denc_vec], 1)#batch * (maxhis + maxsession), max_qdnum + 1, vec
        all_qd_mask = torch.cat([torch.LongTensor(np.ones((all_doc_pos.size()[0], 1))).cuda(cudaid), all_doc_pos], 1)
        
        all_his_vec = torch.sum(all_qdenc_vec, 1)# batch * (maxhis + maxsession), vec
        all_his_vec = all_his_vec.view(-1, self.max_hislen + self.max_sessionlen, self.d_word_vec)
        long_his_vec, short_his_vec = torch.split(all_his_vec, [self.max_hislen, self.max_sessionlen], 1)
        
        #short-long
        short_his_vec = torch.cat([short_his_vec, torch.ones(query.size()[0], 1, self.d_word_vec).cuda(cudaid)], 1)#batch, max_sessionlen + 1, vec
        short_his_vec, *_  = self.encoder_short_his(short_his_vec, short_pos, needpos=True)#batch, max_sessionlen + 1, vec
        short_his, short_user = torch.split(short_his_vec, [self.max_sessionlen, 1], 1)

        long_his_vec = torch.cat([long_his_vec, short_user], 1)#batch, max_hislen + 1, vec
        long_his_vec, *_ = self.encoder_long_his(long_his_vec, long_pos, needpos=True)#batch, max_hislen + 1, vec
        long_his, long_user = torch.split(long_his_vec, [self.max_hislen, 1], 1)#batch, vec
        
        long_user = long_user.squeeze(1)
        short_user = short_user.squeeze(1)

        current_q_long = self.fuse_layer(long_user, current_q_vec)#batch, vec
        current_q_short = self.fuse_layer(short_user, current_q_vec)#batch, vec


        user = self.fuse_layer(long_user, short_user)#batch, vec
        phi = torch.sigmoid(torch.cosine_similarity(user, current_q_vec)).unsqueeze(1)#batch
        
        current_q_final = self.fuse_layer(current_q_long, current_q_short)
        

        #all_current_denc #batch, maxddlen_cur, vec
        all_current_denc_init = all_current_denc.view(-1, self.max_qdnum_cur, self.max_doclen, self.d_word_vec) #batch, max_qdnum_cur, max_doclen, vec

        docs_1_idx = docs1.repeat(self.max_doclen * self.d_word_vec, 1).transpose(0, 1).view(-1, 1, self.max_doclen, self.d_word_vec)#batch, 1, max_doclen, vec
        docs_2_idx = docs2.repeat(self.max_doclen * self.d_word_vec, 1).transpose(0, 1).view(-1, 1, self.max_doclen, self.d_word_vec)#batch, 1, max_doclen, vec
        current_doc_i_1_init = torch.gather(all_current_denc_init, 1, docs_1_idx).squeeze(1)#batch, maxdoclen, vec
        current_doc_i_2_init = torch.gather(all_current_denc_init, 1, docs_2_idx).squeeze(1)#batch, maxdoclen, vec

        current_doc_vec_i_1_init, *_ = self.encoder_term(current_doc_i_1_init, docs1_init)
        current_doc_vec_i_2_init, *_ = self.encoder_term(current_doc_i_2_init, docs2_init)
        current_doc_1_init = torch.sum(current_doc_vec_i_1_init, 1)#batch, vec
        current_doc_2_init = torch.sum(current_doc_vec_i_2_init, 1)#batch, vec


        all_current_denc, *_ = self.encoder_term(all_current_denc, current_d_mask)#batch, maxddlen_cur, vec
        all_current_denc = all_current_denc.view(-1, self.max_qdnum_cur, self.max_doclen, self.d_word_vec)# batch, self.max_qdnum_cur, max_doclen, vec
        
        
        all_current_denc = torch.sum(all_current_denc, 2)# batch, self.max_qdnum_cur, vec
        all_current_click_embedding = self.click_embedding(torch.LongTensor(np.ones((query.size()[0], self.max_qdnum_cur))).cuda(cudaid))# batch, self.max_qdnum_cur, vec
        all_current_denc_vec, *_ = self.encoder_document(all_current_denc + all_current_click_embedding + current_doc_vec_embedding, current_pos, needpos=True)#batch, self.max_qdnum_cur, vec
        docs_1_idx = docs1.repeat(self.d_word_vec, 1).transpose(0, 1).view(-1, 1, self.d_word_vec)#batch, 1, vec
        docs_2_idx = docs2.repeat(self.d_word_vec, 1).transpose(0, 1).view(-1, 1, self.d_word_vec)#batch, 1, vec

        vec_div_score_1, vec_div_score_2 = self.div_layer(all_current_denc_vec, docs1, docs2)#diversified

        #similarity
        query_doc_vec_sim_1 = torch.cosine_similarity(current_q_vec, current_doc_1_init).unsqueeze(1)
        query_doc_vec_sim_2 = torch.cosine_similarity(current_q_vec, current_doc_2_init).unsqueeze(1)

        query_doc_short_sim_1 = torch.cosine_similarity(current_q_short, current_doc_1_init).unsqueeze(1)
        query_doc_short_sim_2 = torch.cosine_similarity(current_q_short, current_doc_2_init).unsqueeze(1)

        query_doc_long_sim_1 = torch.cosine_similarity(current_q_long, current_doc_1_init).unsqueeze(1)
        query_doc_long_sim_2 = torch.cosine_similarity(current_q_long, current_doc_2_init).unsqueeze(1)

        query_doc_final_sim_1 = torch.cosine_similarity(current_q_final, current_doc_1_init).unsqueeze(1)
        query_doc_final_sim_2 = torch.cosine_similarity(current_q_final, current_doc_2_init).unsqueeze(1)

        q_mask = get_non_pad_mask(query)
        d1_mask = get_non_pad_mask(docs1_init)
        d2_mask = get_non_pad_mask(docs2_init)

        interaction_sim_1_init = self.knrm_1(current_q_i, current_doc_i_1_init, q_mask, d1_mask)
        interaction_sim_2_init = self.knrm_1(current_q_i, current_doc_i_2_init, q_mask, d2_mask)

        interaction_sim_1_vec = self.knrm_2(current_q_vec_i, current_doc_vec_i_1_init, q_mask, d1_mask)
        interaction_sim_2_vec = self.knrm_2(current_q_vec_i, current_doc_vec_i_2_init, q_mask, d2_mask)


        feature_sim_1 = self.feature_layer(features1)
        feature_sim_2 = self.feature_layer(features2)

        per_score1_all = torch.cat([query_doc_vec_sim_1, query_doc_short_sim_1, query_doc_long_sim_1, query_doc_final_sim_1, \
                                interaction_sim_1_init, interaction_sim_1_vec, feature_sim_1], 1)
        per_score2_all = torch.cat([query_doc_vec_sim_2, query_doc_short_sim_2, query_doc_long_sim_2, query_doc_final_sim_2, \
                                interaction_sim_2_init, interaction_sim_2_vec, feature_sim_2], 1)
        
        div_score1_all = vec_div_score_1
        div_score2_all = vec_div_score_2

        per_score_1 = self.score_layer_p(per_score1_all)#personal+adhoc
        per_score_2 = self.score_layer_p(per_score2_all)#personal+adhoc
        div_score_1 = self.score_layer_d(div_score1_all)#div
        div_score_2 = self.score_layer_d(div_score2_all)#div
        
        score_1 = phi * per_score_1 + (1 - phi) * div_score_1
        score_2 = phi * per_score_2 + (1 - phi) * div_score_2


        score = torch.cat([score_1, score_2], 1)
        
        p_score = torch.cat([self.pairwise_loss(score_1, score_2),
                    self.pairwise_loss(score_2, score_1)], 1)
        
        '''
        use in the per+div(s) 
        per_p_score = torch.cat([self.pairwise_loss(per_score_1, per_score_2),
                    self.pairwise_loss(per_score_2, per_score_1)], 1)
        per_p_score = torch.softmax(per_p_score, 1)
        per_p_score, _ = torch.split(per_p_score,[1,1],1)
        
        div_p_score = torch.cat([self.pairwise_loss(div_score_1, div_score_2),
                    self.pairwise_loss(div_score_2, div_score_1)], 1)
        div_p_score = torch.softmax(div_p_score, 1)
        div_p_score, _ = torch.split(div_p_score,[1,1],1)
        '''
        pre = F.softmax(score, 1)
        
        return score, pre, p_score