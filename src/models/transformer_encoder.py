"""
Implementation of "Attention is All You Need"
"""
import math
import pdb

import torch.nn as nn
import torch

from models.neural import MultiHeadedAttention
from models.neural import PositionwiseFeedForward
from models.encoder import PositionalEncoding


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, inputs, mask):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm(inputs)
        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)



class myBertSelfAttention(nn.Module):
    def __init__(self):
        super(myBertSelfAttention, self).__init__()
        num_attention_heads=6
        hidden_size=768
        attention_probs_dropout_prob=0.1
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states,key_values, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(key_values)
        mixed_value_layer = self.value(key_values)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:#[16, 20, 5])
            mask = attention_mask.unsqueeze(1).expand_as(attention_scores)#head num
            attention_scores = attention_scores.masked_fill(mask > 0, -1e18)
        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)


        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = context_layer
        return outputs


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout, self.embeddings.embedding_dim)
        self.transformer_local = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src, lengths=None):
        """ See :obj:`EncoderBase.forward()`"""
        batch_size, n_tokens = src.size()
        emb = self.embeddings(src)
        padding_idx = self.embeddings.padding_idx
        mask_hier = 1 - src.data.eq(padding_idx).int()
        out = self.pos_emb(emb)

        for i in range(self.num_layers):
            out = self.transformer_local[i](out, out, 1 - mask_hier)  # all_sents * max_tokens * dim
        out = self.layer_norm(out)
        mask_hier = mask_hier[:, :, None].float()
        src_features = out * mask_hier
        return src_features, mask_hier


class EMEncoder(nn.Module):
    def __init__(self, args, device):
        super(EMEncoder, self).__init__()
        self.args = args
        self.device = device
        self.padding_idx = 0

        # Para attends to the clusters
        # CLuster attends to the paras
        self.E2A = myBertSelfAttention()
        self.R2A = myBertSelfAttention()
        self.E2R = myBertSelfAttention()

        self.A2R = myBertSelfAttention()
        self.R2E = myBertSelfAttention()
        self.A2E = myBertSelfAttention()

        self.E2E = myBertSelfAttention()
        self.R2R = myBertSelfAttention()

        self.fusion1 = nn.Linear(self.args.enc_hidden_size * 3, self.args.enc_hidden_size)
        self.fusion2 = nn.Linear(self.args.enc_hidden_size * 3, self.args.enc_hidden_size)
        self.fusion3 = nn.Linear(self.args.enc_hidden_size * 2, self.args.enc_hidden_size)
        self.self_attn_layer_norm = nn.LayerNorm(self.args.enc_hidden_size)

        self.condition = nn.Linear(5, 5)
        self.relu = nn.Sigmoid()

    def forward(self, abs_words, abs_node, mask_src, ref_node, ref_words, mask_refs, ref_entity, key_words, key_node):
        abs_node = abs_node.unsqueeze(1)  # [16,1,768]
        mask_node_refs = mask_refs.view(self.args.batch_size, 5, self.args.max_pos)
        mask_node_refs = mask_node_refs.sum(2)  # 16,5
        mask_node_refs = mask_node_refs >= 1  # 16,5
        mask_node_refs = mask_node_refs.type(torch.long)
        reverse_ref_entity = ref_entity.transpose(1, 2).contiguous()

        for i in range(2):
            key_node1= self.R2E( key_values=ref_node, hidden_states=key_node, attention_mask=1-reverse_ref_entity)  # [16, 5, 20]
            key_node2 = self.A2E(key_values=abs_node,  hidden_states=key_node)
            key_node3 = self.E2E(key_values=key_node, hidden_states=key_node)
            key_node = self.fusion1(torch.cat([key_node1, key_node2, key_node3], -1))  # [16, 20, 768]
            key_node = self.self_attn_layer_norm(key_node)

            self_ref_mask=torch.matmul(mask_node_refs.unsqueeze(-1).float(),mask_node_refs.unsqueeze(1).float())
            ref_node1= self.R2R(key_values=ref_node,hidden_states=ref_node,
                                   attention_mask=1-self_ref_mask)
            ref_node2 = self.E2R(hidden_states=ref_node, key_values=key_node,  attention_mask=1-ref_entity)  # ([16, 40, 768])

            ref_node3 = self.A2R(key_values=ref_node, hidden_states=ref_node,
                                  attention_mask=1-self_ref_mask)  # ([16, 5, 768])
            sim = torch.matmul(ref_node, abs_node.transpose(1, 2)).squeeze(-1)  # [16,5]
            sim = self.relu(self.condition(sim)).unsqueeze(-1)  # [16,5,1]
            sim = sim.repeat(1, 1, self.args.enc_hidden_size)
            ref_node3 = ref_node3.mul(sim) # ([16, 5, 768])
            ref_node = self.fusion2(torch.cat([ref_node1, ref_node2, ref_node3], -1))  # [16, 5, 768]
            ref_node = self.self_attn_layer_norm(ref_node)

            abs_node1 = self.R2A(key_values=ref_node,  hidden_states=abs_node,
                                  attention_mask=1-mask_node_refs.unsqueeze(1))  # ([16, 1, 768])
            abs_node2 = self.E2A(key_values=key_node,  hidden_states=abs_node)
            abs_node = self.fusion3(torch.cat([abs_node1, abs_node2], -1))  # [16, 1, 768]
            abs_node = self.self_attn_layer_norm(abs_node)

        ref_node_words = torch.unsqueeze(ref_node, 2)  # ([16, 5, 1,768]
        ref_context = ref_words + ref_node_words  # [16, 5, 200, 768]
        ref_context = self.self_attn_layer_norm(ref_context)
        ref_context=ref_context.view(self.args.batch_size,-1,self.args.enc_hidden_size) # [16, 5*200, 768]

        abs_context = abs_words + abs_node
        abs_context = self.self_attn_layer_norm(abs_context)

        key_node_words = torch.unsqueeze(key_node, 2)  # ([16, 20, 1,768]
        key_context = key_words + key_node_words
        key_context = self.self_attn_layer_norm(key_context)
        key_context=key_context.view(self.args.batch_size,-1,self.args.enc_hidden_size)#[16,40,768]
        return abs_context,ref_context,key_context,ref_words,mask_node_refs



