#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math
import pdb

import torch

from others.utils import rouge_results_to_str, test_rouge, tile
from translate.beam import GNMTGlobalScorer
import numpy as np

def build_predictor_my(args, tokenizer, symbols, model, logger=None):
    scorer = GNMTGlobalScorer(args.alpha, length_penalty='wu')

    translator = Translator(args, model, tokenizer, symbols, global_scorer=scorer, logger=logger)
    return translator


class Translator(object):
    def __init__(self,
                 args,
                 model,
                 vocab,
                 symbols,
                 global_scorer=None,
                 logger=None,
                 dump_beam=""):
        self.logger = logger
        self.cuda = args.visible_gpus != '-1'

        self.args = args
        self.model = model
        self.generator = self.model.generator
        self.vocab = vocab
        self.symbols = symbols
        self.start_token = symbols['BOS']
        self.end_token = symbols['EOS']

        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length

        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert (len(translation_batch["gold_score"]) ==
                len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, gold_score, tgt_str, src_str = translation_batch["predictions"], translation_batch["scores"], \
                                                      translation_batch["gold_score"], batch.tgt_str, batch.src_str

        translations = []
        for b in range(batch_size):
            pred_sents = self.vocab.convert_ids_to_tokens([int(n) for n in preds[b][0]])
            pred_sents = ' '.join(pred_sents).replace(' ##', '')
            gold_sent = ' '.join(tgt_str[b].split())
            raw_src = ' '.join(src_str[b])
            translation = (pred_sents, gold_sent, raw_src)
            translations.append(translation)

        return translations

    def translate(self,
                  data_iter, step,
                  attn_debug=False):

        self.model.eval()
        gold_path = self.args.result_path + '.%d.gold' % step
        can_path = self.args.result_path + '.%d.candidate' % step
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')

        # raw_gold_path = self.args.result_path + '.%d.raw_gold' % step
        # raw_can_path = self.args.result_path + '.%d.raw_candidate' % step
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')

        raw_src_path = self.args.result_path + '.%d.raw_src' % step
        self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')

        # pred_results, gold_results = [], []
        ct = 0
        with torch.no_grad():
            for batch in data_iter:
                print(str(ct))
                if (self.args.recall_eval):
                    gold_tgt_len = batch.tgt.size(1)
                    self.min_length = gold_tgt_len + 20
                    self.max_length = gold_tgt_len + 60
                batch_data = self.translate_batch(batch,ct)
                translations = self.from_batch(batch_data)

                for trans in translations:
                    pred, gold, src = trans
                    pred_str = pred.replace('[unused0]', '').replace('[unused3]', '').replace('[PAD]', '').replace(
                        '[unused1]', '').replace(r' +', ' ').replace(' [unused2] ', '<q>').replace('[unused2]',
                                                                                                '').strip()

                    gold_str = gold.strip()
                    if (self.args.recall_eval):
                        _pred_str = ''
                        gap = 1e3
                        for sent in pred_str.split('<q>'):
                            can_pred_str = _pred_str + '<q>' + sent.strip()
                            can_gap = math.fabs(len(_pred_str.split()) - len(gold_str.split()))
                            # if(can_gap>=gap):
                            if (len(can_pred_str.split()) >= len(gold_str.split()) + 10):
                                pred_str = _pred_str
                                break
                            else:
                                gap = can_gap
                                _pred_str = can_pred_str

                    self.can_out_file.write(pred_str + '\n')
                    self.gold_out_file.write(gold_str + '\n')
                    self.src_out_file.write(src + '\n')
                    ct += 1
                self.can_out_file.flush()
                self.gold_out_file.flush()
                self.src_out_file.flush()

        self.can_out_file.close()
        self.gold_out_file.close()
        self.src_out_file.close()

        if (step != -1):
            rouges = self._report_rouge(gold_path, can_path)
            self.logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
            f = open('rouges/' + self.args.exp_name, 'a')
            f.write('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
            f.close()

    def _report_rouge(self, gold_path, can_path):
        self.logger.info("Calculating Rouge")
        results_dict = test_rouge(self.args.temp_dir, can_path, gold_path)
        return results_dict

    def translate_batch(self, batch,ct):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            return self._fast_translate_batch(
                batch,
                self.max_length,ct,
                min_length=self.min_length)

    def _fast_translate_batch(self,
                              batch,
                              max_length,ct,
                              min_length=0):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert not self.dump_beam

        beam_size = self.beam_size
        batch_size = batch.batch_size
        src = batch.src
        segs = batch.segs
        mask_src = batch.mask_src

        refs = batch.refs
        mask_refs = batch.mask_refs
        ref_entity = batch.ref_entity
        keyphrases = batch.keyphrases
        refs_segs = batch.refs_segs

        abs_words = self.model.bert(src, segs, mask_src)
        abs_doc = torch.mean(abs_words, 1)

        refs = refs.view(self.args.batch_size * 5, -1)
        mask_sent_refs = mask_refs.view(self.args.batch_size * 5, -1)
        refs_segs = refs_segs.view(self.args.batch_size * 5, -1)
        ref_words = self.model.bert(refs, refs_segs, mask_sent_refs)

        ref_words = ref_words.view(self.args.batch_size, 5, self.args.max_pos, self.args.enc_hidden_size)
        ref_doc = torch.mean(ref_words, 2)  # [16,5,768] check mask?

        keyphrases = keyphrases.view(self.args.batch_size * 20, -1)
        key_words = self.model.bert(keyphrases, torch.ones_like(keyphrases), torch.ones_like(keyphrases))
        key_words = key_words.view(self.args.batch_size, 20, 2, self.args.enc_hidden_size)
        key_one = torch.mean(key_words, 2)  # [16,20,768]

        if self.args.model=='graph' or self.args.model=='hier' or 'full' in self.args.model:
            abs_context, ref_context, key_context, ref_words, mask_doc_refs = self.model.Graphencoder(abs_words, abs_doc,
                                                                                                      mask_src, ref_doc,
                                                                                                      ref_words, mask_refs,
                                                                                                      ref_entity,
                                                                                                      key_words,
                                                                                                      key_one)
        elif self.args.model == 'baseline' or self.args.model=='contras':
            abs_context = abs_words
            ref_context = ref_words
            key_context = key_words

        refs = refs.view(self.args.batch_size, -1)  # [16,5*200]
        keyphrases = keyphrases.view(self.args.batch_size, -1)

        dec_states1 = self.model.decoder.init_decoder_state(keyphrases, key_context, with_cache=True)
        dec_states2 = self.model.decoder.init_decoder_state(src, abs_context, with_cache=True)
        dec_states3 = self.model.decoder.init_decoder_state(refs, ref_context, with_cache=True)

        device = abs_context.device

        # Tile states and memory beam_size times.
        dec_states1.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        dec_states2.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        dec_states3.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        abs_context = tile(abs_context, beam_size, dim=0)
        ref_context = tile(ref_context, beam_size, dim=0)
        key_context = tile(key_context, beam_size, dim=0)

        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            self.start_token,
            dtype=torch.long,
            device=device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch


        content_all=[]

        for step in range(max_length):
            decoder_input = alive_seq[:, -1].view(1, -1)

            # Decoder forward.
            decoder_input = decoder_input.transpose(0, 1)

            dec_out, dec_states1, dec_states2, dec_states3 ,attn= self.model.decoder(decoder_input,key_context, dec_states1, abs_context, dec_states2,
                                                                                ref_context, dec_states3,
                                                                                 step=step)

            # Generator forward.
            log_probs = self.generator.forward(dec_out.transpose(0, 1).squeeze(0))
            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, self.end_token] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            if (self.args.block_trigram):
                cur_len = alive_seq.size(1)
                if (cur_len > 3):
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        words = self.vocab.convert_ids_to_tokens(words)
                        words = ' '.join(words).replace(' ##', '').split()
                        if (len(words) <= 3):
                            continue
                        trigrams = [(words[i - 1], words[i], words[i + 1]) for i in range(1, len(words) - 1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = -10e20

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            select_indices=select_indices.type(torch.LongTensor).cuda()
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():

                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]

                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
            # Reorder states.
            select_indices = batch_index.view(-1)
            select_indices=select_indices.type(torch.LongTensor).cuda()
            abs_context = abs_context.index_select(0, select_indices)
            ref_context = ref_context.index_select(0, select_indices)
            key_context = key_context.index_select(0, select_indices)


            dec_states1.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))
            dec_states2.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))
            dec_states3.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))
            # exit()

        return results

