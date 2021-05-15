import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ElectraModel, BertModel, BertTokenizer, BertConfig
from my_transformer import MultiHeadAttention, PositionWiseFeedForward, SinusoidalPositionalEncodedEmbedding

def masked_cross_entropy_for_value(logits, target, pad_idx=0):
    '''
    Args:
        logits (Tensor): shape (batch_size, J, max_len, vocab_size)
        target (Tensor): shape (batch_size*J*max_len). 각 배치의 각 slot에 대한 예측 value 토큰들이 쫙 나열된 상태. (batch_size, J, max_len).view(-1)
    '''
    mask = target.ne(pad_idx) ## target에서 값이 0인 부분은 True로, 나머지는 False로 이루어진 벡터. target과 같은 shape을 가진다.
    logits_flat = logits.view(-1, logits.size(-1)) # shape (batch_size*J*max_len, vocab_size)
    log_probs_flat = torch.log(logits_flat) # shape (batch_size*J*max_len, vocab_size)
    target_flat = target.view(-1, 1) # shape (batch_size*J*max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat) # losses_flat[i, 0] = log_probs_flat[i, target_flat[i,0]]
    ## target_flat[i,0]은 i번째 토큰의 실제 인덱스. log_probs_flat[i, target_flat[i,0]]은 i번째 토큰의 예측 score(정확히는 score에 log를 씌운 것.) --> log_probs_flat[i, target_flat[i,0]]이 loss가 된다.
    losses = losses_flat.view(*target.size()) # shape (batch_size*J*max_len)
    losses = losses * mask.float() ## 패딩 토큰 부분의 loss는 0이된다.
    loss = losses.sum() / (mask.sum().float()) ## mask.sum() = 패딩토큰이 아닌 것들의 개수 
    return loss


class TRADE(nn.Module):
    def __init__(self, config, tokenized_slot_meta, pad_idx=0):
        super(TRADE, self).__init__()
        # self.encoder = GRUEncoder(
        #     config.vocab_size,
        #     config.hidden_size,
        #     1,
        #     config.hidden_dropout_prob,
        #     config.proj_dim,
        #     pad_idx,
        # )
        if config.model_name_or_path:
            self.encoder = BertModel.from_pretrained(config.model_name_or_path)
        else:
            self.encoder = BertModel(config)

        self.decoder = TransformerDecoder(config)

        self.decoder.set_slot_idx(tokenized_slot_meta) ## domain-slot을 토크나이징 하여 만든 인덱스들에 패딩까지 넣는다(가장 긴 길이를 기준으로 패딩). 
        self.tie_weight()
        
    def set_subword_embedding(self, model_name_or_path):
        model = ElectraModel.from_pretrained(model_name_or_path)
        self.encoder.embed.weight = model.embeddings.word_embeddings.weight
        self.tie_weight()

    def tie_weight(self):
        # self.decoder.embed.weight = self.encoder.embed.weight
        # if self.decoder.proj_layer:
        #     self.decoder.proj_layer.weight = self.encoder.proj_layer.weight
        self.decoder.embed.weight = self.encoder.embeddings.word_embeddings.weight

    def forward(
        self, 
        input_ids,
        target_ids,
        token_type_ids=None, 
        attention_mask=None,    
        ):
        '''
        Args:
            input_ids
            target_ids
            token_type_ids
            attention_mask
        '''

        encoder_outputs, pooled_output = self.encoder(input_ids=input_ids, 
                                                      token_type_ids=token_type_ids, 
                                                      attention_mask=attention_mask)
        ## (batch_size, seq_len, hidden_size), (batch_size, hidden_size)
        all_point_outputs, all_gate_outputs = self.decoder(
            input_ids,
            target_ids,
            encoder_outputs,
            attention_mask,
        )

        return all_point_outputs, all_gate_outputs
    
    def predict(self,
                input_ids,
                token_type_ids=None, 
                attention_mask=None, 
                ):
        '''
        Args:
            src (LongTensor): input to encoder. shape '(batch_size, src_len)'
        
        Returns:
            output_tokens (LongTensor): predicted tokens. shape'(batch_size, max_position)'
        '''
#         padding_mask = self.generate_padding_mask(src)
        
        encoder_outputs, pooled_output = self.encoder(input_ids=input_ids, 
                                                      token_type_ids=token_type_ids, 
                                                      attention_mask=attention_mask)
        
        
        output_tokens = (torch.ones((self.config.batch_size, self.config.max_position))\
                         * self.TRG.vocab.stoi['<pad>']).long().to(self.device) 
        ## (batch_size, max_position)
        output_tokens[:,0] = self.TRG.vocab.stoi['<sos>']
        for trg_index in range(1, self.config.max_position):
            trg = output_tokens[:,:trg_index] # (batch_size, trg_index)
            causal_mask = self.generate_causal_mask(trg) # (trg_index, trg_index)
            output, _ = self.decoder(input_indices = trg,
                                     encoder_output = encoder_output,
                                     enc_dec_attention_padding_mask = padding_mask,
                                     causal_mask = causal_mask) # (batch_size, trg_index, emb_dim)
            output = self.linear(output) # (batch_size, trg_index, # trg vocab)
            output = torch.argmax(output, dim = -1) # (batch_size, trg_index)
            output_tokens[:,trg_index] = output[:,-1]
        
        return output_tokens

class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, dropout, proj_dim=None, pad_idx=0):
        super(GRUEncoder, self).__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        if proj_dim:
            self.proj_layer = nn.Linear(d_model, proj_dim, bias=False)
        else:
            self.proj_layer = None

        self.d_model = proj_dim if proj_dim else d_model
        self.gru = nn.GRU(
            self.d_model,
            self.d_model,
            n_layer,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        mask = input_ids.eq(self.pad_idx).unsqueeze(-1) ## torch.Size([B, seq_len, 1])
        '''
        input_ids[0] = tensor([    2,     3, 11655,  4279,  8553,    18,  6265, 10806,  4073,  8117,
         4070,  6259,  4279,  4219,    16, 10472,  4110,  6477,  4279,  4034,
        20762,  4403,  4292,  6722,  4076,  8553,    18,     3, 11655,  4279,
         8553,    18,  3201, 29365,  4034, 27672,  4070,  3249,  4154,  8553,
           35,     3, 27672,  4034, 14053,  4576,  6216,    18,     3,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0],device='cuda:0')
        input_ids.eq(self.pad_idx)[0] = tensor([False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True],device='cuda:0')
        '''
        x = self.embed(input_ids) ## torch.Size([batch_size, seq_len, hidden_size])
        if self.proj_layer:
            x = self.proj_layer(x)
        x = self.dropout(x)
        o, h = self.gru(x)
        '''
        o.szie() = torch.Size([batch_size, 258, hidden_size * 2]) foward와 backward를 이어서 출력하므로 hidden_size * 2이다.
        h.size() = (2, batch_size, hidden_size). 모든 배치에 대해서 foward의 마지막과 backward의 마지막을 출력함.
        '''
        o = o.masked_fill(mask, 0.0) 
        '''패딩토큰에 해당하는 부분을 마스킹. 패딩 토큰에 해당하는 부분의 (hidden_size * 2크기의)벡터의 값을 전부 0으로 만듦. 나중에 p^history를 만들 때 패딩토큰에 대한 output은 영향력이 없도록 만들기 위한 것 같음.'''
        output = o[:, :, : self.d_model] + o[:, :, self.d_model :] ## (batch_size, seq_len, hidden_size)
        hidden = h[0] + h[1]  # foward의 마지막과 backward의 마지막을 더함. shape (batch_size, hidden_size)
        return output, hidden


class SlotGenerator(nn.Module):
    def __init__(
        self, vocab_size, hidden_size, dropout, n_gate, proj_dim=None, pad_idx=0
    ):
        super(SlotGenerator, self).__init__()
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_idx
        )  # shared with encoder

        if proj_dim:
            self.proj_layer = nn.Linear(hidden_size, proj_dim, bias=False)
        else:
            self.proj_layer = None
        self.hidden_size = proj_dim if proj_dim else hidden_size

        self.gru = nn.GRU(
            self.hidden_size, self.hidden_size, 1, dropout=dropout, batch_first=True
        )
        self.n_gate = n_gate
        self.dropout = nn.Dropout(dropout)
        self.w_gen = nn.Linear(self.hidden_size * 3, 1) ## p^gen계산에 쓰이는 W_1임.
        self.sigmoid = nn.Sigmoid()
        self.w_gate = nn.Linear(self.hidden_size, n_gate) ## G_j 계산에 쓰이는 W_g임.

    def set_slot_idx(self, slot_vocab_idx):
        whole = []
        max_length = max(map(len, slot_vocab_idx))
        for idx in slot_vocab_idx:
            if len(idx) < max_length:
                gap = max_length - len(idx)
                idx.extend([self.pad_idx] * gap)
            whole.append(idx)
        self.slot_embed_idx = whole  # torch.LongTensor(whole)

    def embedding(self, x):
        x = self.embed(x)
        if self.proj_layer:
            x = self.proj_layer(x)
        return x

    def forward(
        self, input_ids, encoder_output, hidden, input_masks, max_len, teacher=None
    ):
        '''
        Args:
            input_ids (Tensor) : shape (batch_size, seq_len)
            encoder_output (Tensor) : shape (batch_size, seq_len, hidden_size)
            hidden (Tensor) : shape (1, batch_size, hidden_size)
            input_masks (Tensor) : shape (batch_size, seq_len)
            max_len (Int) : train시에는 target_ids.size(-1)로 주고 inference는 특정 값으로 고정하여 준다.
            teacher (Tensor) : train시에는 특정 확률로 target_id(batch_size, num_slot, target_id 토큰갯수) 이거나 None임. inference시에는 None.
        '''
        input_masks = input_masks.ne(1) ## input_masks의 True와 False를 반전시킨다.(True는 False로, False는 True로)
        # J, slot_meta : key : [domain, slot] ex> LongTensor([1,2])
        # J,2
        batch_size = encoder_output.size(0)
        slot = torch.LongTensor(self.slot_embed_idx).to(input_ids.device) 
        '''domain-slot을 토크나이징하여 얻은 인덱스들에 패딩까지 넣은것이 slot임
        shape (J, 4). 현재 데이터에서는 4가 토크나이징된 domain-slot들 중에 가장 긴 것이다.
        '''
        slot_e = torch.sum(self.embedding(slot), 1)  # (J, embedding_dim = 768)
        '''self.embedding(slot).size() = torch.Size([J, slot_length, hidden_size])
        한 domain-slot에 대한 모든 토큰들의 임베딩벡터를 합친다.
        --> slot_e[0]은 0번째 slot의 embedding vector인 것임.
        '''
        J = slot_e.size(0)

        all_point_outputs = torch.zeros(batch_size, J, max_len, self.vocab_size).to(
            input_ids.device
        ) 
        ''' torch.Size([4, 45, 5, 35000]). 각 decoding step에 대해서 35000개의 vocab에 대한 distribution이 나오게 됨. (batch_size, J, max_decoding_step, vocab_size)'''
        # Parallel Decoding -> 모든 slot에 대한 디코딩을 동시에 진행한다.
        w = slot_e.repeat(batch_size, 1).unsqueeze(1) ## 디코더의 첫번째 스텝에서의 input으로 쓰임.
        '''shape (J*batch_size, 1, hidden_size). (J, hidden_size)가 똑같이 batch_size개수 만큼 있게 됨.
        '''
        hidden = hidden.repeat_interleave(J, dim=1) ## 디코더의 initial hidden state로 쓰인다.
        '''shape (1, J*batch_size, hidden_size)
        1차원 방향으로 첫번째 data에 대한 hidden_size크기의 벡터가 똑같은게 J개 있고 그다음
        두번째 data에 대한 hidden_size크기의 벡터가 똑같은게 J개 있고 ...
        그다음 batch_size번째 data에 대한 hidden_size크기의 벡터가 똑같은게 J개 있음.
        '''
        encoder_output = encoder_output.repeat_interleave(J, dim=0)
        '''(J*batch_size, seq_len, hidden_size)첫번째 데이터에 대한 (1, seq_len, hidden_size)가 J번 반복되고 
        그 다음 두번째 데이터에 대한 (1, seq_len, hidden_size)가 J번 반복되고 ...
        '''
        input_ids = input_ids.repeat_interleave(J, dim=0)
        '''(J*batch_size, seq_len)
        첫번째 데이터에 대한 input_id가 J번 반복되고 두번째 데이터에 대한 input_id가 J번 반복되고 ...
        '''
        input_masks = input_masks.repeat_interleave(J, dim=0)
        '''(J*batch_size, seq_len)
        첫번째 데이터에 대한 input_mask가 J번 반복되고 두번째 데이터에 대한 input_mask가 J번 반복되고 ...
        
        '''
        for k in range(max_len): ## max_len번 만큼 디코딩한다. max_len = train시에는 target_ids.size(-1).
            w = self.dropout(w)
            _, hidden = self.gru(w, hidden)  # 1,B,D (1,J*batch_size, hidden_size)

            ### p^history 계산. B,T,D * B,D,1 => B,T
            attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # B,T,1
            '''(J*batch_size, seq_len, hidden_size)와 (J*batch_size, hidden_size, 1)을 배치 매트릭스곱을 한다. -> (J*batch_size, seq_len, 1)
            '''
            attn_e = attn_e.squeeze(-1).masked_fill(input_masks, -1e9)
            '''항상 softmax에 넣기전에 패딩토큰 부분을 마스킹해야 한다. (J*batch_size, seq_len)
            '''
            attn_history = F.softmax(attn_e, -1)  # B,T
            
            
            ### p^vocab 계산.
            if self.proj_layer:
                hidden_proj = torch.matmul(hidden, self.proj_layer.weight)
            else:
                hidden_proj = hidden

            # B,D * D,V => B,V
            attn_v = torch.matmul(
                hidden_proj.squeeze(0), self.embed.weight.transpose(0, 1)
            ) 
            '''(J*batch_size, hidden_size)와 (hidden_size, vocab_size)를 매트릭스곱을 함. --> (J*batch_size, vocab_size)
            '''
            attn_vocab = F.softmax(attn_v, -1)

            # P^gen 계산.  B,1,T * B,T,D => B,1,D
            context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # c_jk 계산
            ''' (J*batch_size, 1, seq_len)와 (J*batch_size, seq_len, hidden_size)를 행렬곱
                --> (J*batch_size, 1, hidden_size)
            '''
            p_gen = self.sigmoid(
                self.w_gen(torch.cat([w, hidden.transpose(0, 1), context], -1))
            )  
            ''' (J*batch_size, 1, 1)'''
            p_gen = p_gen.squeeze(-1) 
            ''' (J*batch_size, 1) '''

            p_context_ptr = torch.zeros_like(attn_vocab).to(input_ids.device) 
            '''(J*batch_size, vocab_size)'''
            p_context_ptr.scatter_add_(1, input_ids, attn_history)  # copy B,V
            '''
            p_context_ptr[i][input_ids[i][j]] += attn_history[i][j], 0<=i<=J*batch_size, 
            0<=j<=seq_len.
            --> (J*batch_size, vocab_size)
            '''
            p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # B,V
            '''(J*batch_size, vocab_size). 각 데이터의 각 slot에 대한 k번째 디코딩 스텝의 확률분포.'''
            _, w_idx = p_final.max(-1) ## shapne [J*batch_size,]

            if teacher is not None:
                '''teacher -> (batch_size, num_slot, target_id 토큰갯수)'''
                w = self.embedding(teacher[:, :, k]).reshape(batch_size * J, 1, -1)
                # rand_idx = torch.randint(5, 34000, w_idx.size()).to(input_ids.device) 
                # w = self.embedding(rand_idx).unsqueeze(1)
                '''self.embedding(teacher[:, :, k]) = 모든 배치데이터의 모든 slot에 대한 value의 k번째 토큰의 embedidng vector(batch_size, num_slot, emb_dim).
                '''
            else:
                w = self.embedding(w_idx).unsqueeze(1)  
                '''(J*batch_size, 1 ,hidden_size)'''
            if k == 0: 
                '''첫번째 스텝에서만 slot_gate값을 계산한다.'''
                gated_logit = self.w_gate(context.squeeze(1))  # (J*batch_size, 3)
                all_gate_outputs = gated_logit.view(batch_size, J, self.n_gate) ## 모든 slot에 대한 gate값 예측.
            all_point_outputs[:, :, k, :] = p_final.view(batch_size, J, self.vocab_size) ## 모든 slot에 대한 각 decoding step의 vocab 확률분포 예측.

        return all_point_outputs, all_gate_outputs
    
class DecoderLayer(nn.Module):
    def __init__(self, 
                 config):
        '''Initialize decoder layer
        
        Args:
            config (Config): configuration parameters.
        '''
        
        super().__init__()
        self.hidden_dropout_prob = config.hidden_dropout_prob
        
        # masked multi_head attention
        self.self_attn = MultiHeadAttention(emb_dim = config.hidden_size,
                                            num_heads = config.num_attention_heads,
                                            drop_out = config.attention_drop_out,
                                            causal = True)
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)
        
        # encoder-decoder attention
        self.enc_dec_attn = MultiHeadAttention(emb_dim = config.hidden_size,
                                               num_heads = config.num_attention_heads,
                                               drop_out = config.attention_drop_out,
                                               encoder_decoder_attention = True)
        self.enc_dec_attn_layer_norm = nn.LayerNorm(config.hidden_size)
        
        #position-wise feed forward
        self.position_wise_feed_forward = PositionWiseFeedForward(config.hidden_size,
                                                               config.ffn_dim,
                                                               config.drop_out)
        self.feed_forward_layer_norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self,
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                enc_dec_attention_padding_mask: torch.Tensor = None,
                causal_mask: torch.Tensor = None):
        
        '''
        Args:
            x (Tensor): Input to decoder layer. shape '(batch_size, trg_len, emb_dim)'.
            encoder_output (Tensor): Output of encoder. shape '(batch_size, src_len, emb_dim)'
            enc_dec_attention_padding_mask (Tensor): Binary BoolTensor for masking padding of
                                                     encoder output.
                                                     shape '(batch_size, src_len)'.
            causal_mask (Tensor): Binary BoolTensor for masking future information in decoder.
                                  shape '(batch_size, trg_len)'
        
        Returns:
            x (Tensor): Output of decoder layer. shape '(batch_size, trg_len, emb_dim)'.
            self_attn_weights (Tensor): Masked self attention weights of decoder. 
                                        shape '(batch_size, trg_len, trg_len)'.
            enc_dec_attn_weights (Tensor): Encoder-decoder attention weights.
                                           shape '(batch_size, trg_len, src_len)'.
        '''
        
        # msked self attention
        residual = x
        x, self_attn_weights = self.self_attn(query = x,
                                              key = x,
                                              attention_mask = causal_mask)
        x = F.dropout(x, p = self.hidden_dropout_prob, training = self.training)
        x = self.self_attn_layer_norm(x + residual)
        
        # encoder-decoder attention
        residual = x
        x, enc_dec_attn_weights = self.enc_dec_attn(query = x,
                                                    key = encoder_output,
                                                    attention_mask = enc_dec_attention_padding_mask)
        x = F.dropout(x, p = self.hidden_dropout_prob, training = self.training)
        x = self.enc_dec_attn_layer_norm(x + residual)
        
        # position-wise feed forward
        residual = x
        x = self.position_wise_feed_forward(x)
        x = self.feed_forward_layer_norm(x + residual)
        
        return x, self_attn_weights, enc_dec_attn_weights
    
class TransformerDecoder(nn.Module):
    
    def __init__(self, 
                 config, pad_idx=0):
        '''Initialize stack of Decoder layers
        
        Args:
            config (Config):Configuration parameters.

        '''
        
        super().__init__()
        
        self.pad_idx = pad_idx
        self.vocab_size = config.vocab_size
        self.embed = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=pad_idx
        )  # shared with encoder

        if proj_dim:
            self.proj_layer = nn.Linear(config.hidden_size, config.proj_dim, bias=False)
        else:
            self.proj_layer = None
        self.hidden_size = config.proj_dim if config.proj_dim else config.hidden_size

#         self.gru = nn.GRU(
#             self.hidden_size, self.hidden_size, 1, dropout=dropout, batch_first=True
#         )
        self.n_gate = config.n_gate
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.w_gen = nn.Linear(self.hidden_size * 3, 1) ## p^gen계산에 쓰이는 W_1임.
        self.sigmoid = nn.Sigmoid()
        self.w_gate = nn.Linear(self.hidden_size, config.n_gate) ## G_j 계산에 쓰이는 W_g임.
        
        ########################기존 트랜스포머 코드들
        self.hidden_dropout_prob = config.hidden_dropout_prob
        
        self.embed_positions = SinusoidalPositionalEncodedEmbedding(config.max_position,
                                                                    hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_decoder_layers)])
        
    def set_slot_idx(self, slot_vocab_idx):
        whole = []
        max_length = max(map(len, slot_vocab_idx))
        for idx in slot_vocab_idx:
            if len(idx) < max_length:
                gap = max_length - len(idx)
                idx.extend([self.pad_idx] * gap)
            whole.append(idx)
        self.slot_embed_idx = whole
    
    def embedding(self, x):
        x = self.embed(x)
        if self.proj_layer:
            x = self.proj_layer(x)
        return x
    
    def generate_causal_mask(self,  
                             trg: torch.LongTensor):
        '''Generate padding mask and causal mask
        
        Args:
            trg (LongTensor): input to decoder. shape '(batch_size, trg_len)'
        
        Returns:
            causal_mask (Tensor): shape '(trg_len, trg_len)'
        '''
        tmp = torch.ones(trg.size(1)+1, trg.size(1)+1, dtype = torch.bool)
        '''dom_slot을 고려하여 trg.size(1)+1처럼 1을 더해준다.
        '''
        causal_mask = torch.tril(tmp,-1).transpose(0,1).contiguous().to(self.device)
        
        return causal_mask

# padding mask는 decoder에는 필요없다.
#     def generate_padding_mask(self, 
#                               src: torch.LongTensor):
#         '''Generate padding mask
        
#         Args:
#             src (LongTensor): input to encoder. shape '(batch_size, src_len)'
        
#         Returns:
#             padding_mask (Tensor): shape '(batch_size, src_len)'
#         '''
#         padding_mask = src.eq(self.SRC.vocab.stoi['<pad>']).to(self.device)
        
#         return padding_mask
    
    
    
    def forward(self,
                input_ids: torch.Tensor,
                target_ids: torch.Tensor,
                encoder_output: torch.Tensor,
                input_masks: torch.Tensor = None,):
        '''
        Args:
            input_ids (Tensor): dialogue context token ids. shape '(batch_size, src_len)'
            target_ids (Tensor): input to decoder. shape '(batch_size, J = num_slot, trg_len)'
            encoder_output (Tensor): output of encoder. shape '(batch_size, src_len, emb_dim)'
            input_masks (Tensor): Binary BoolTensor for masking padding of encoder output. 인코더에서 쓴 패딩마스크임.shape '(batch_size, src_len)'.

        
        Returns:
            x (Tensor): output of decoder. shape '(batch_size, trg_len, emb_dim)'
            enc_dec_attn_weigths (list): list of enc-dec attention weights of each Decoder layer.
        '''
        
        # dom_slot자리를 고려하여 causal_mask를 한칸더 크게 만들기 - 했음.
        causal_mask = self.generate_causal_mask(target_ids) # shape (trg_len+1, trg_len+1)
        ## causal_mask는 parallel decoding을 위한 처리를 따로 할 필요없다. 알아서 브로드캐스팅 됨. MultiHeadAttention의 코드 참고.
        

        #####################################
        input_masks = input_masks.ne(1) ## input_masks의 True와 False를 반전시킨다.(True는 False로, False는 True로)
'''transformer 라이브러리의 BERT에서 쓰는 패딩마스크는 [True, True, True, True,True, False, False, False] 와 같은 형태임. 내가 짠 transformer코드에서는 [False, False, False, False, False, True, True, True]같은 형태를 쓰므로 토글시켜줘야 한다.'''
        # J, slot_meta : key : [domain, slot] ex> LongTensor([1,2])
        # J,2
        
        slot = torch.LongTensor(self.slot_embed_idx).to(input_ids.device) 
        '''domain-slot을 토크나이징하여 얻은 인덱스들에 패딩까지 넣은것이 slot임
        shape (J, 4). 현재 데이터에서는 4가 토크나이징된 domain-slot들 중에 가장 긴 것이다.
        '''
        slot_e = torch.sum(self.embedding(slot), 1)  # (J, embedding_dim = 768)
        '''self.embedding(slot).size() = torch.Size([J, slot_length, hidden_size])
        한 domain-slot에 대한 모든 토큰들의 임베딩벡터를 합친다. -> (J, hidden_size)
        --> slot_e[0]은 0번째 slot의 embedding vector인 것임.
        '''
        batch_size = encoder_output.size(0)
        J, hidden_size = slot_e.size()
        trg_len = target_ids.size(-1)
        ##inputs_embed에 dom_slot에 대한 embeding vecotor를 맨 앞에 넣어준다.(J*batch_size, trg_len+1, hidden_size)로 만들어야 함.
        decoder_input = torch.embty(J*batch_size, trg_len+1, hidden_size) # (J*batch_size, trg_len+1, hidden_size)
        slot_e = slot_e.repeat(batch_size, 1) # (J*batch_size, hidden_size)
        target_ids = target_ids.reshape(J*batch_size, -1) ## (J*batch_size, trg_len)
        targets_embed = self.embed(target_ids) ## (J*batch_size, trg_len, hidden_size)
        
        decoder_input[:,1:,:] = targets_embed
        decoder_input[:,0,:] = slot_e
        '''첫번째 데이터에 대한 slot1 의 value벡터(trg_len, hidden_size)가 나오고 그다음
        첫번째 데이터에 대한 slot2의 value벡터(trg_len, hidden_size)가 나오고...
        하는 식으로 첫번째 데이터에 대한 J개 slot에 대한 value벡터 (J, trg_len, hidden_size)
        가 나옴. 그 뒤로는 두번째 데이터에 대한 J개의 slot에 대한 value벡터 (J, trg_len, hidden_size)
        가 나오는 식임.
        '''
        ##dom_slot을 고려하여 한칸 만큼 더 큰 pos_embed를 만든다. -> 했음.
        pos_embed = self.embed_positions(target_ids) ## (J*batch_size, trg_len+1, hidden_size)
        decoder_input = decoder_input + pos_embed
        decoder_input = F.dropout(x, p = self.hidden_dropout_prob, training = self.training)
        
        encoder_output = encoder_output.repeat_interleave(J, dim=0)
        '''(J*batch_size, src_len, hidden_size)첫번째 데이터에 대한 (1, src_len, hidden_size)가 J번 반복되고 
        그 다음 두번째 데이터에 대한 (1, seq_len, hidden_size)가 J번 반복되고 ...
        '''
        
        enc_dec_attn_weights = []
        for decoder_layer in self.layers:
            decoder_input, _, attn_weights = decoder_layer(decoder_input, 
                                                           encoder_output,
                                                           enc_dec_attention_padding_mask,
                                                           causal_mask)
            '''decoder_input shape (J*batch_size, trg_len+1, hidden_size)
               attn_weights shape (J*batch_size, # attn head, trg_len+1, src_len)
            '''
        decoder_output = decoder_input # (J*batch_size, trg_len+1, hidden_size)

        all_point_outputs = torch.zeros(batch_size, J, max_len, self.vocab_size).to(
            input_ids.device
        ) 
        ''' torch.Size([4, 45, 5, 35000]). 각 decoding step에 대해서 35000개의 vocab에 대한 distribution이 나오게 됨. (batch_size, J, max_decoding_step, vocab_size)'''
        # Parallel Decoding -> 모든 slot에 대한 디코딩을 동시에 진행한다.
        

        input_ids = input_ids.repeat_interleave(J, dim=0)
        '''(J*batch_size, src_len)
        첫번째 데이터에 대한 input_id가 J번 반복되고 두번째 데이터에 대한 input_id가 J번 반복되고 ...
        '''
        input_masks = input_masks.repeat_interleave(J, dim=0)
        '''(J*batch_size, src_len)
        첫번째 데이터에 대한 input_mask가 J번 반복되고 두번째 데이터에 대한 input_mask가 J번 반복되고 ... '''
        attn_e = torch.bmm(decoder_output, encoder_output.transpose(-1,-2))
        '''(J*batch_size, trg_len+1, hidden_size) x (J*batch_size, hidden_size, src_len)
        -> (J*batch_size, trg_len+1, src_len)
        '''
#         attn_e = attn_weights.sum(dim=1) ## (J*batch_size, trg_len+1, src_len)
#         attn_e = p_history.sum(dim=1) ## (J*batch_size, src_len)
        attn_e = attn_e.masked_fill(input_masks.unsqueeze(1), -1e9)  ## (J*batch_size, trg_len+1, src_len)
        attn_history = F.softmax(attn_e, -1)  ## (J*batch_size, trg_len+1, src_len)
        
        attn_v = torch.matmul(decoder_output, self.embed.weight.transpose(0, 1))
        '''(J*batch_size, trg_len+1, hidden_size) x (hidden_size, vocab_size)
        -> (J*batch_size, trg_len+1, vocab_size)
        '''
        attn_vocab = F.softmax(attn_v, -1)
        
        p_gen = self.sigmoid(
                self.w_gen(decoder_output)
            ) # (J*batch_size, trg_len+1, 1)
        
        p_context_ptr = torch.zeros_like(attn_vocab).to(input_ids.device)
        ## (J*batch_size, trg_len+1, vocab_size)
        p_context_ptr.scatter_add_(2, input_ids.unsqueeze(1).repeat(1,trg_len+1,1), attn_history)
        '''attn_history shape (J*batch_size, trg_len+1, src_len).
        input_ids.unsqueeze(1).repeat(1,trg_len+1,1) shape -> (J*batch_size, trg_len+1, src_len).
        input_ids.unsqueeze(1).repeat(1,trg_len+1,1)[0] 은 input_ids[0,:]가 trg_len+1번 반복되어 있음.
        
        p_context_ptr[i][j][input_ids[i][j][k]] += attn_history[i][j][k]
        '''
        
        p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr
        '''attn_vocab shape (J*batch_size, trg_len+1, vocab_size)
        p_context_ptr shape (J*batch_size, trg_len+1, vocab_size)
        p_gen shape = (J*batch_size, trg_len+1, 1)
        '''
        all_point_outputs = p_final.view(batch_size, J, trg_len+1, -1)
        ## (batch_size, J, trg_len+1, vocab_size)
        
        gated_logit = self.w_gate(decoder_output[:,0,:])
        '''decoder_output[:,0,:] shape -> (J*batch_size, hidden_size)
        gated_logit shape -> (J*batch_size, n_gate)
        '''
        all_gate_outputs = gated_logit.view(batch_size, J, self.n_gate)
        ## (batch_size, J, n_gate)
        return all_point_outputs, all_gate_outputs
        
