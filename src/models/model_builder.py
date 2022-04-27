import copy
import torch.nn.functional as F

import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_
from models.encoder import Classifier, ExtTransformerEncoder
from models.optimizers import Optimizer
from models.transformer_encoder import TransformerEncoder, EMEncoder
from models.mydecoder_my_weight import myTransformerDecoder_weight


def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='fix',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    return optim


def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)

    return optim


def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)

    return optim


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator


class CNN_Text(nn.Module):
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        D = 50
        Ci = 1
        kernel_num = 1
        kernel_size = 5
        self.convs = nn.Conv2d(Ci, kernel_num, (kernel_size, D))
        self.dropout = nn.Dropout(args.enc_dropout)
        self.fc1 = nn.Linear(kernel_num, 1)

    def forward(self, x):  # (B, S, D)
        x = x.unsqueeze(1)  # (B, Ci, S, D)
        x = F.relu(self.convs(x)).mean(3)  # [(N, Co, W), ...]*len(Ks)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # [(N, Co), ...]*len(Ks)
        x = self.dropout(x).squeeze(-1)  # (B,10)
        return x

def local_mutual_info(args, CNN_Text, ref_words, nega_token_ref_words, decoder_outputs, mask_refs, nega_mask_refs):
    ground_truth_label = torch.ones(args.batch_size, dtype=torch.long,
                                    requires_grad=False).cuda()
    nega_truth_label = torch.zeros(args.batch_size, dtype=torch.long,
                                   requires_grad=False).cuda()

    ref_context = ref_words[:, 0, :, :]  # [16, 200, 768]
    generate = decoder_outputs.repeat(1, args.max_pos, 1)  # [16, 200, 768]
    positive_input = torch.cat([generate, ref_context], 2)  # [16, 200, 768*2]
    positive = CNN_Text(positive_input)  ##  [16,1]

    neg_context = nega_token_ref_words[:, 0, :, :]  # [16, 200, 768]
    negative_input = torch.cat([generate, neg_context], 2)  # [16, 200, 768*2]
    negative = CNN_Text(negative_input)  ##[16,1]
    label = torch.cat([ground_truth_label, nega_truth_label], -1)
    predict = torch.cat([positive, negative], -1)

    return label, predict


class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if (large):
            self.model = BertModel.from_pretrained('bert-large-uncased')
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased')

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if (self.finetune):
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)
        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
                                     num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads,
                                     intermediate_size=args.ext_ff_size)
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)

        if (args.max_pos > 512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,
                                                  :].repeat(args.max_pos - 512, 1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

        if self.args.model == 'graph' or self.args.model == 'hier' or 'full' in self.args.model:
            self.Graphencoder = EMEncoder(self.args, self.device)
            if args.mode == 'train':
                module_state_dict = self.bert.model.encoder.layer[-1].attention.self.state_dict()
                self.Graphencoder.E2A.load_state_dict(module_state_dict)
                self.Graphencoder.R2A.load_state_dict(module_state_dict)
                self.Graphencoder.E2R.load_state_dict(module_state_dict)
                self.Graphencoder.A2R.load_state_dict(module_state_dict)
                self.Graphencoder.R2E.load_state_dict(module_state_dict)
                self.Graphencoder.A2E.load_state_dict(module_state_dict)
                self.Graphencoder.E2E.load_state_dict(module_state_dict)
                self.Graphencoder.R2R.load_state_dict(module_state_dict)

        if (args.max_pos > 512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,
                                                  :].repeat(args.max_pos - 512, 1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        if args.model == 'full':
            self.decoder = myTransformerDecoder_weight(
                self.args.dec_layers,
                self.args.dec_hidden_size, heads=self.args.dec_heads,
                d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings,
                max_pos=self.args.max_pos)
            self.CNN_Text = CNN_Text(self.args)
            self.global_contra = nn.Linear(self.args.enc_hidden_size * 2, 1)
            self.relu = nn.ReLU()

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if (args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, mask_src, mask_tgt, refs, mask_refs, refs_segs,
                ref_entity, keyphrases, nega_ids, nega_segs, mask_nega):
        # initialization
        abs_words = self.bert(src, segs, mask_src)
        abs_node = torch.mean(abs_words, 1)

        refs = refs.view(self.args.batch_size * 5, -1)
        mask_sent_refs = mask_refs.view(self.args.batch_size * 5, -1)
        refs_segs = refs_segs.view(self.args.batch_size * 5, -1)
        ref_words = self.bert(refs, refs_segs, mask_sent_refs)
        ref_words = ref_words.view(self.args.batch_size, 5, self.args.max_pos, self.args.enc_hidden_size)
        ref_node = torch.mean(ref_words, 2)  # [16,5,768] check mask?

        keyphrases = keyphrases.view(self.args.batch_size * 20, -1)
        key_words = self.bert(keyphrases, torch.ones_like(keyphrases), torch.ones_like(keyphrases))
        key_words = key_words.view(self.args.batch_size, 20, 2, self.args.enc_hidden_size)
        key_node = torch.mean(key_words, 2)  # [16,20,768]

        if 'full' in self.args.model:
            abs_context, ref_context, key_context, ref_words, mask_doc_refs = self.Graphencoder(abs_words, abs_node,
                                                                                                mask_src, ref_node,
                                                                                                ref_words, mask_refs,
                                                                                                ref_entity, key_words,
                                                                                                key_node)


        refs = refs.view(self.args.batch_size, -1)  # [16,5*200]
        keyphrases = keyphrases.view(self.args.batch_size, -1)

        dec_state1 = self.decoder.init_decoder_state(keyphrases, key_context)
        dec_state2 = self.decoder.init_decoder_state(src, abs_context)  # [16,200]  [16,200,768]
        dec_state3 = self.decoder.init_decoder_state(refs, ref_context)  # [16,5*200,768]
        decoder_outputs, _, _, _, _ = self.decoder(tgt[:, :-1], key_context, dec_state1,
                                                   abs_context, dec_state2, ref_context, dec_state3,
                                                   )
        label = None
        local_predict = None
        global_predict = None

        if  'full' in self.args.model:
            nega_ids = nega_ids.view(self.args.batch_size * 5, -1)
            nega_segs = nega_segs.view(self.args.batch_size * 5, -1)
            mask_nega = mask_nega.view(self.args.batch_size * 5, -1)
            nega_words = self.bert(nega_ids, mask_nega, nega_segs)
            mask_doc_nega = mask_nega.view(self.args.batch_size, 5, self.args.max_pos)
            mask_doc_nega = mask_doc_nega.sum(2)  # 16,5
            mask_doc_nega = mask_doc_nega >= 1  # 16,5
            mask_doc_nega = mask_doc_nega.type(torch.long)
            nega_words = nega_words.view(self.args.batch_size, 5, self.args.max_pos, self.args.enc_hidden_size)
            generate = decoder_outputs[:, -1, :].unsqueeze(1)  # [16, 1, 768]

            # local
            label, local_predict = local_mutual_info(
                self.args, self.CNN_Text,
                ref_words,
                nega_words,
                generate, mask_doc_refs,
                mask_doc_nega)
            # global
            ref_words = torch.mean(torch.mean(ref_words, 1), 1)  # [16,768]
            nega_words = torch.mean(torch.mean(nega_words, 1), 1)  # [16,768]
            generate = generate.squeeze(1)  # [16, 768]
            global_positive = self.global_contra(torch.cat([generate, ref_words], 1)).squeeze(-1)
            global_negative = self.global_contra(torch.cat([generate, nega_words], 1)).squeeze(-1)
            global_predict = torch.cat([global_positive, global_negative], -1)

        return decoder_outputs, None, label, local_predict, global_predict
