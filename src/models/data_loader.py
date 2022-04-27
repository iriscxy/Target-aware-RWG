import bisect
import gc
import glob
import pdb
import random

import torch

from others.logging import logger


class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, data=None, device=None, is_test=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            pre_src = [x[0] for x in data]
            pre_tgt = [x[1] for x in data]
            keyphrases = [x[4] for x in data]
            refs = [x[5] for x in data]
            ref_entity = [x[6] for x in data]
            segments_ids = [x[7] for x in data]
            refs_segs = [x[8] for x in data]
            nega_ids = [x[9] for x in data]
            nega_segs = [x[10] for x in data]

            refs_segs = torch.tensor(refs_segs)
            nega_segs = torch.tensor(nega_segs)
            nega_ids = torch.tensor(nega_ids)
            refs = torch.tensor(refs)
            ref_entity = torch.stack(ref_entity, 0)
            keyphrases = torch.stack(keyphrases, 0)
            segments_ids = torch.tensor(self._pad(segments_ids, 0))

            src = torch.tensor(self._pad(pre_src, 0))
            tgt = torch.tensor(self._pad(pre_tgt, 0))

            mask_src = 1 - (src == 0).int().to(torch.long)
            mask_tgt = 1 - (tgt == 0).int().to(torch.long)
            mask_refs = 1 - (refs == 0).int().to(torch.long)
            mask_nega = 1 - (nega_ids == 0).int().to(torch.long)

            setattr(self, 'src', src.to(device))
            setattr(self, 'tgt', tgt.to(device))
            setattr(self, 'mask_src', mask_src.to(device))
            setattr(self, 'mask_tgt', mask_tgt.to(device))
            setattr(self, 'segs', segments_ids.to(device))
            setattr(self, 'refs_segs', refs_segs.to(device))
            setattr(self, 'mask_nega', mask_nega.to(device))
            setattr(self, 'nega_segs', nega_segs.to(device))
            setattr(self, 'nega_ids', nega_ids.to(device))

            setattr(self, 'mask_refs', mask_refs.to(device))
            setattr(self, 'refs', refs.to(device))
            setattr(self, 'ref_entity', ref_entity.to(device))
            setattr(self, 'keyphrases', keyphrases.to(device))

            src_str = [x[2] for x in data]
            setattr(self, 'src_str', src_str)
            tgt_str = [x[3] for x in data]
            setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        return self.batch_size


def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """

    # assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pt = args.bert_data_path
    yield _lazy_dataset_loader(pt, corpus_type)


def abs_batch_size_fn(new, count):
    src, tgt = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(tgt))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    if (count > 6):
        return src_elements + 1e3
    return src_elements


def ext_batch_size_fn(new, count):
    if (len(new) == 4):
        pass
    src, labels = new[0], new[4]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets, batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args=self.args,
                            dataset=self.cur_dataset, batch_size=self.batch_size,
                            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset, batch_size, device=None, is_test=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0
        if (self.args.task == 'abs'):
            self.batch_size_fn = abs_batch_size_fn
        else:
            self.batch_size_fn = ext_batch_size_fn

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        src = ex['src']
        keyphrase_tokens = ex['keyphrase_tokens']
        refs_ids = ex['refs_ids']

        ref_sent_entity = ex['ref_sent_entity']

        tgt = ex['tgt'][:self.args.max_tgt_len][:-1] + [2]

        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']

        segments_ids = ex['segs']
        refs_segs = ex['refs_segs']

        segments_ids = segments_ids[:self.args.max_pos]
        end_id = [src[-1]]

        src = src[:-1][:self.args.max_pos - 1] + end_id
        refs_ids = [
            each[:self.args.max_pos - 1] + end_id if each[self.args.max_pos - 1] != 0 else each[:self.args.max_pos] for
            each in
            refs_ids]
        refs_segs = [each[:self.args.max_pos] for each in refs_segs]

        nega_segs = ex['nega_segs']
        nega_ids = ex['nega_ids']

        nega_ids = [
            each[:self.args.max_pos - 1] + end_id if each[self.args.max_pos - 1] != 0 else each[:self.args.max_pos] for
            each in
            nega_ids]
        nega_segs = [each[:self.args.max_pos] for each in nega_segs]

        return src, tgt, src_txt, tgt_txt, keyphrase_tokens, refs_ids, ref_sent_entity, \
               segments_ids, refs_segs, nega_ids, nega_segs

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if (len(ex['src']) == 0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if (ex is None):
                continue
            minibatch.append(ex)
            if len(minibatch) == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0

    def batch(self, data, batch_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size):
            p_batch = list(buffer)
            yield p_batch

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)

                yield batch
            return
