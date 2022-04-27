import gc
import glob
import hashlib
import itertools
import json
import os
import random
import re
from tqdm import tqdm

import torch
from transformers import  BertTokenizer, BertModel


class BertData():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def token_sent(self, ref):
        ref = ref[:40]
        ref = [each.split() for each in ref]
        ref_txt = [' '.join(sent) for sent in ref]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(ref_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        while len(src_subtoken_idxs) < 200:
            src_subtoken_idxs.append(0)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        while len(segments_ids) < 200:
            segments_ids.append(0)
        return src_subtoken_idxs, segments_ids

    def preprocess(self, src, tgt, refs, keyphrases, negative):
        tgt = [each.split() for each in tgt]
        keyphrase_tokens = self.tokenizer(keyphrases, max_length=2, padding='max_length', truncation=True,
                                          add_special_tokens=False, return_tensors='pt')['input_ids']
        ref_sent_entity = torch.zeros(5, 20)
        for key_index, keyphrase in enumerate(keyphrases):
            for ref_index, ref in enumerate(refs):
                ref_str = ' '.join(ref)
                if [ele for ele in keyphrase.split() if (ele in ref_str)]:
                    ref_sent_entity[ref_index, key_index] = torch.tensor(1)
        ref_sent_entity = ref_sent_entity.type(torch.long)

        refs_ids = []
        refs_segs = []
        for ref in refs:
            src_subtoken_idxs, segments_ids = self.token_sent(ref)
            refs_ids.append(src_subtoken_idxs)
            refs_segs.append(segments_ids)

        padding = [0 for _ in range(200)]
        while len(refs_ids) < 5:
            refs_ids.append(padding)
        while len(refs_segs) < 5:
            refs_segs.append(padding)

        nega_ids = []
        nega_segs = []
        for ref in negative:
            src_subtoken_idxs, segments_ids = self.token_sent(ref)
            nega_ids.append(src_subtoken_idxs)
            nega_segs.append(segments_ids)
        while len(nega_ids) < 5:
            nega_ids.append(padding)
        while len(nega_segs) < 5:
            nega_segs.append(padding)

        src_subtoken_idxs, segments_ids = self.token_sent(src)

        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt))) for tt in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:200]

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = '<q>'.join([' '.join(tt) for tt in src])
        return src_subtoken_idxs, tgt_subtoken_idxs, src_txt, tgt_txt, keyphrase_tokens, refs_ids, ref_sent_entity, \
               segments_ids, refs_segs, nega_ids, nega_segs


def _format_to_bert(input, output):
    datasets = []
    bert = BertData()
    f = open(input)
    save_file = output
    lines = f.readlines()
    for line in tqdm(lines[:]):
        content = json.loads(line)
        source, tgt = content['abstract'], content['relatedwork']
        refs = content['refs']
        refs = [each['abstract'] for each in refs]
        refs = refs[:5]

        negative = content['negative']
        negative = [each['abstract'] for each in negative]
        negative = negative[:5]

        keyphrases = content['keywords']
        b_data = bert.preprocess(source, tgt, refs, keyphrases, negative)
        if b_data is not None:
            src_subtoken_idxs, tgt_subtoken_idxs, src_txt, tgt_txt, \
            keyphrase_tokens, refs_ids, ref_sent_entity, \
            segments_ids, refs_segs, nega_ids, nega_segs = b_data
            b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                           "keyphrase_tokens": keyphrase_tokens,
                           "refs_ids": refs_ids,
                           "segs": segments_ids,
                           "refs_segs": refs_segs,
                           "nega_ids": nega_ids,
                           "nega_segs": nega_segs,
                           "ref_sent_entity": ref_sent_entity,
                           'src_txt': src_txt, "tgt_txt": tgt_txt}
            datasets.append(b_data_dict)
    print(str(len(datasets)))
    torch.save(datasets, save_file)


_format_to_bert(input='train_check_contras.json',
                output='train.pt')
