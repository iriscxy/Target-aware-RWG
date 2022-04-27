import os
import argparse
import json
from tqdm import tqdm
import random
f = open('test_check.json')
f_train = open('train_check.json')

ref_num = 5

fout = open('test_check_contras.json', 'w')

lines = f.readlines()
nega_lines = f_train.readlines()

for line in tqdm(lines):
    content=json.loads(line)
    random_num=random.randint(0,len(nega_lines)-1)
    nega_line=nega_lines[random_num]
    nega_content=json.loads(nega_line)
    content['negative']=nega_content['refs']
    json.dump(content, fout)
    fout.write("\n")
