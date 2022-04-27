"""
Implementation of "Attention is All You Need"
"""
import json
import pdb

import torch
import torch.nn as nn
import numpy as np

from models.encoder import PositionalEncoding
from models.neural import MultiHeadedAttention, PositionwiseFeedForward, DecoderState, MultiHeadedAttention_weight

MAX_SIZE = 5000


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the PositionwiseFeedForward.
      dropout (float): dropout probability(0-1.0).
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout, max_pos):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)

        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)

    def forward(self, inputs, memory_bank1, memory_bank2, memory_bank3, src_pad_mask1, \
                src_pad_mask2, src_pad_mask3, tgt_pad_mask,
                key_abs_similarity, key_refs_similarity,
                previous_input=None, layer_cache1=None, layer_cache2=None, layer_cache3=None, step=None):
        dec_mask = torch.gt(tgt_pad_mask +
                            self.mask[:, :tgt_pad_mask.size(1),
                            :tgt_pad_mask.size(1)], 0)
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None

        query, _ = self.self_attn(all_input, all_input, input_norm,
                                  mask=dec_mask,
                                  layer_cache=layer_cache1,
                                  type="self")
        query = self.drop(query) + inputs
        query_norm = self.layer_norm_2(query)
        mid1, drop_key_attn = self.context_attn(memory_bank1, memory_bank1, query_norm,
                                                mask=src_pad_mask1,
                                                layer_cache=layer_cache1,
                                                type="context")  # [16, 8, 149, 40]
        mid2, key_abs_attn = self.context_attn(memory_bank2, memory_bank2, query_norm,
                                               mask=src_pad_mask2,
                                               layer_cache=layer_cache2,
                                               type="context")  # [16, 40, 200]
        mid3, key_refs_attn = self.context_attn(memory_bank3, memory_bank3, query_norm,
                                                mask=src_pad_mask3,
                                                layer_cache=layer_cache3,
                                                type="context")
        mid4, key_abs_attn = self.context_attn(memory_bank2, memory_bank2, mid1,
                                               mask=src_pad_mask2,
                                               layer_cache=layer_cache2,
                                               type="context")  # [16, 40, 200]
        mid5, key_refs_attn = self.context_attn(memory_bank3, memory_bank3, mid1,
                                                mask=src_pad_mask3,
                                                layer_cache=layer_cache3,
                                                type="context")
        output = self.feed_forward(
            self.drop(mid1) + self.drop(mid2) + self.drop(mid3) + self.drop(mid4) + self.drop(mid5) + query)

        return output, all_input
        # return output, all_input,content

        # return output

    def _get_attn_subsequent_mask(self, size):
        """
        Get an attention mask to avoid using the subsequent info.

        Args:
            size: int

        Returns:
            (`LongTensor`):

            * subsequent_mask `[1 x size x size]`
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class myTransformerDecoder_weight(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings, max_pos):
        super(myTransformerDecoder_weight, self).__init__()

        # Basic attributes.
        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout, self.embeddings.embedding_dim)

        # Build TransformerDecoder.
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout, max_pos)
             for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tgt, memory_bank1, state1, memory_bank2, state2, memory_bank3, state3, step=None):
        src_words1 = state1.src
        src_words2 = state2.src
        src_words3 = state3.src

        tgt_words = tgt
        src_batch1, src_len1 = src_words1.size()
        src_batch2, src_len2 = src_words2.size()
        src_batch3, src_len3 = src_words3.size()

        tgt_batch, tgt_len = tgt_words.size()

        # Run the forward pass of the TransformerDecoder.
        # emb = self.embeddings(tgt, step=step)
        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = self.pos_emb(emb, step)

        src_memory_bank1 = memory_bank1
        src_memory_bank2 = memory_bank2
        src_memory_bank3 = memory_bank3

        padding_idx = self.embeddings.padding_idx
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)
        src_pad_mask1 = src_words1.data.eq(padding_idx).unsqueeze(1) \
            .expand(src_batch1, tgt_len, src_len1)
        src_pad_mask2 = src_words2.data.eq(padding_idx).unsqueeze(1) \
            .expand(src_batch2, tgt_len, src_len2)
        src_pad_mask3 = src_words3.data.eq(padding_idx).unsqueeze(1) \
            .expand(src_batch3, tgt_len, src_len3)
        if state1.cache is None:
            saved_inputs = []
        key_abs_similarity = torch.matmul(memory_bank1, memory_bank2.transpose(1, 2))
        key_refs_similarity = torch.matmul(memory_bank1, memory_bank3.transpose(1, 2))
        for i in range(self.num_layers):
            prev_layer_input = None
            if state1.cache is None:
                if state1.previous_input is not None:
                    prev_layer_input = state1.previous_layer_inputs[i]
            output, all_input \
                = self.transformer_layers[i](
                output, src_memory_bank1, src_memory_bank2, src_memory_bank3,
                src_pad_mask1, src_pad_mask2, src_pad_mask3, tgt_pad_mask,
                key_abs_similarity, key_refs_similarity,
                previous_input=prev_layer_input,
                layer_cache1=state1.cache["layer_{}".format(i)]
                if state1.cache is not None else None,
                layer_cache2=state2.cache["layer_{}".format(i)]
                if state2.cache is not None else None,
                layer_cache3=state3.cache["layer_{}".format(i)]
                if state3.cache is not None else None,
                step=step)

            if state1.cache is None:
                saved_inputs.append(all_input)

        if state1.cache is None:
            saved_inputs = torch.stack(saved_inputs)

        output = self.layer_norm(output)

        # Process the result and update the attentions.

        if state1.cache is None:
            state1 = state1.update_state(tgt, saved_inputs)
        if state2.cache is None:
            state2 = state2.update_state(tgt, saved_inputs)
        if state3.cache is None:
            state3 = state3.update_state(tgt, saved_inputs)
        return output, state1, state2, state3, None

    def init_decoder_state(self, src, memory_bank,
                           with_cache=False):
        """ Init decoder state """
        state = TransformerDecoderState(src)
        if with_cache:
            state._init_cache(memory_bank, self.num_layers)
        return state


class TransformerDecoderState(DecoderState):
    """ Transformer Decoder state base class """

    def __init__(self, src):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None
        self.cache = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        if (self.previous_input is not None
                and self.previous_layer_inputs is not None):
            return (self.previous_input,
                    self.previous_layer_inputs,
                    self.src)
        else:
            return (self.src,)

    def detach(self):
        if self.previous_input is not None:
            self.previous_input = self.previous_input.detach()
        if self.previous_layer_inputs is not None:
            self.previous_layer_inputs = self.previous_layer_inputs.detach()
        self.src = self.src.detach()

    def update_state(self, new_input, previous_layer_inputs):
        state = TransformerDecoderState(self.src)
        state.previous_input = new_input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def _init_cache(self, memory_bank, num_layers):
        self.cache = {}

        for l in range(num_layers):
            layer_cache = {
                "memory_keys": None,
                "memory_values": None
            }
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            self.cache["layer_{}".format(l)] = layer_cache

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = self.src.data.repeat(1, beam_size, 1)

    def map_batch_fn(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.src = fn(self.src, 0)
        if self.cache is not None:
            _recursive_map(self.cache)
