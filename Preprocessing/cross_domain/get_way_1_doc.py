import json
import pickle
from collections import Counter
import math
import nltk
from tqdm import tqdm
import random
import jsonlines
from itertools import combinations
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import word_tokenize
import spacy

prepath = '../../../project_data/DocEE/cross_setting/' 
nlp = spacy.load("en_core_web_sm")

split = 'train' # train, dev, test
N_way = 6 # get preprocessed files for constructing support, query pairs

with open( prepath + split + '_new.pkl', 'rb') as f:
    train = json.load(f)

new_train = {}
for x in train:
    if x[2] not in new_train:
        new_train[x[2]] = []
    else:
        new_train[x[2]].append(x)

with open('arguments_event.json', 'r') as f:
    arguments_event = json.load(f) # event1: [arg1, arg2]

with open( 'event_split.json', 'r') as f:
    event_split = json.load( f)

train_events, dev_events, test_events = event_split['train_events'], event_split['dev_events'], event_split['test_events']

def sample_1_doc(train_events, new_train, way, doc):
    sampling_res = {}
    for event in tqdm( train_events ) :
        print(event)
        sampling_res[event] = {}
        sampled = []
        count, count1 = 0, 0

        while count1 < len(new_train[event])*500 and count < 50000:
            count1 += 1
            i = random.randint(0, len(new_train[event]) - 1 )

            if (i) in sampled:
                continue

            sampled.append((i))
            event_id = list(set([x['type'] for x in json.loads(new_train[event][i][3])]))

            if len(set(event_id)) >= way:
                if tuple(set(event_id)) not in sampling_res[event]:
                    sampling_res[event][tuple(set(event_id))] = []
                sampling_res[event][tuple(set(event_id))].append([i])
                count += 1
        print("########")
        print(count)
        
    return sampling_res

if split == 'train':
    sampling_2_args = sample_1_doc(train_events, new_train, way = N_way, doc = 1 )
elif split == 'dev':
    sampling_2_args = sample_1_doc(dev_events, new_train, way = N_way, doc = 1 )
else:
    sampling_2_args = sample_1_doc(test_events, new_train, way = N_way, doc = 1 )

with open( prepath + '6w1d/'+ split + '_com_6w1d.pkl', 'wb') as f:
    pickle.dump(sampling_2_args, f)

# python get_way_1_doc.py