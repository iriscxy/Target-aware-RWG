import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import pdb
import json
import sys
from tqdm import tqdm
from keybert import KeyBERT

kw_model = KeyBERT()
f = open('train.json')
fw = open('train_check.json', 'w')
ref_num = 5

lines = f.readlines()
for line in tqdm(lines):
    content = json.loads(line)
    relatedwork = content['relatedwork']
    relatedwork = ' '.join(relatedwork)
    abstract = content['abstract']
    abstract = ' '.join(abstract)
    doc = ' '.join(abstract.split()[:200])
    refs = content['refs']
    refs = [each for each in refs if len(each) != 0]
    refs=refs[:ref_num]
    refs = [each['abstract'] for each in refs]
    refs = [' '.join(each) for each in refs]
    refs=[' '.join(ref.split()[:200]) for ref in refs]
    refs = ' '.join(refs)
    doc += refs
    keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(2, 2), stop_words='english',
                                         top_n=20)
    select_keywords=[word[0] for word in keywords]
    content['keywords'] = select_keywords

    json.dump(content, fw)
    fw.write('\n')
