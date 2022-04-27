# Target-aware-RWG
Official code and dataset for SIGIR 2022 paper 'Target-aware Abstractive Related Work Generation with Contrastive Learning'.


# About the corpus
TAS2 corpus consists of 107,700 reference-relatedwork training pairs, 5000 validation and 5000 test pairs.
TAD corpus consists of 208,255 reference-relatedwork training pairs, 5000 validation and 5000 test pairs.


# How to get these datasets?
Signed the following copyright announcement with your name and organization. Then complete the form online(https://forms.gle/51HSxUHcpui7k4Ep6) and mail to xiuying.chen#kaust.edu.sa ('#'->'@'), we will send you the corpus by e-mail when approved.

# Copyright
The original copyright belongs to the source owner.

The copyright of annotation belongs to our group, and they are free to the public.

The dataset is only for research purposes. Without permission, it may not be used for any commercial purposes and distributed to others.

# Preprocess
The process for the two datasets are similar, and here we use the TAD dataset as an example.
First, unzip all the data files into a directory.
Go to `prepro/tad` directory, run `1keyphrases.py`, `2contras.py`, and `3topt.py`, respectively, to obtain keyphrases and negative contrastive cases as input, and finally convert the json files to `.pt` files.

# Train & Evaluate

Before training, you should install all the required packages listed in the `requirements.txt` file.
Finally, go into the source code directory, and run:

```shell script
python train.py  -task abs -mode train -bert_data_path data_to/train.pt -dec_dropout 0.2 -sep_optim false -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 8000 -batch_size 16 -train_steps 400000 -report_every 500 -accum_count 1 -use_bert_emb true -use_interval true  -exp_name tag -max_tgt_len 150 -max_length 150 -visible_gpus 0 -finetune false -lr 0.0001 -min_length 100
```


After training, the checkpoints can be found at `model_path/args.exp_name`.
This code will automatically evaluate the model in every 8000 training steps the evaluation ROUGE scores of the test dataset are listed in file `rouges/args.exp_name`, and the decoding results are in `results/args.exp_name`.
The frequency of automatic model evaluation can be changed using flag `--save_checkpoint_steps` (default value is 8000 steps).


# Citation
We appreciate your citation if you find our dataset is beneficial.

```
@inproceedings{chen2022target,
  title={Target-aware Abstractive Related Work Generation with Contrastive Learning},
  author={Xiuying Chen, Hind Alamro, Mingzhe Li, Shen Gao, Rui Yan, Xin Gao and Xiangliang Zhang},
  booktitle = {SIGIR},
  year = {2022}
}
```