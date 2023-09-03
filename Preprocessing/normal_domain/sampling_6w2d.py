import json
import pickle
from tqdm import tqdm
import random
import jsonlines, pickle
import spacy

prepath = '../../../project_data/DocEE/normal_setting/6w2d/'
nlp = spacy.load("en_core_web_sm")

split = 'train' # 'train, 'dev', 'test'
if split != 'train':
    lengh_control = 5000
else:
    lengh_control = 15 # control sampling length

############ tokenizations #############
def tokenizations(dataset):
    sentence = dataset[1]
    spans = []
    doc = nlp(sentence)
    for token in doc:
        spans.append([token.idx, token.idx + len(token)])

    labels = ['O'] * len(spans)
    tokens = [sentence[x[0]: x[1]] for x in spans]
    argument_list = dataset[3]
    res = json.loads(argument_list)
    #label = list(set([x['type'] for x in res]))
    for i in range(len(labels)):
        for entity in res:
            start = entity['start']
            end = entity['end']
            if spans[i][0] >= start and spans[i][1] <= end + 1:
                labels[i] = entity['type']
    return tokens, labels  # tokens, labels, label: set(label1, ...)

############ open files #############
with open( 'event_split.json', 'r') as f:
    event_split = json.load( f)

with open( prepath[:-5] + split + '_new.pkl', 'rb') as f:
    train = json.load(f)

new_train = {}
for x in train:
    if x[2] not in new_train:
        new_train[x[2]] = []
    else:
        new_train[x[2]].append(x)

train_events = event_split[ split + '_events']

with open( prepath + split + '_com_6w2d.pkl', 'rb') as f:
    sampling_2_args = pickle.load( f ) 

############ open files #############
def get_event_key_instance(new_train, event, key, comb_list):

    sampling_res = {}
    for p1 in comb_list:
        d1 = new_train[event][p1[0]]
        tokens1, labels1 = tokenizations(d1)

        d2 = new_train[event][p1[1]]
        tokens2, labels2 = tokenizations(d2)

        picked_comb = key # random choose 6 args if there are more than 6 args types
        for i, tok in enumerate(labels1):
            if tok != 'O' and tok not in picked_comb:
                labels1[i] = 'O'

        for i, tok in enumerate(labels2):
            if tok != 'O' and tok not in picked_comb:
                labels2[i] = 'O'

        if set(labels1 + labels2) == set(picked_comb + ['O']) and len(set(labels1 + labels2)) == 7:
            word = [tokens1, tokens2]
            label = [labels1, labels2]
            instance = {'word': word, 'label': label }
            sampling_res[tuple(p1)] = instance
        if len(sampling_res) > lengh_control:
            break

    unique_pairs = []
    for key1 in sampling_res:
        for key2 in sampling_res:
            if set(key1).intersection(key2) == set():
                if [key1, key2] not in unique_pairs and [key2, key1] not in unique_pairs:
                    unique_pairs.append([key1, key2])
        if len(unique_pairs) > lengh_control:
            break
    # unique pairs for [support, query]
    random.shuffle(unique_pairs)
    if len(unique_pairs) > lengh_control:
        chocie = [unique_pairs[x] for x in random.sample(range(len(unique_pairs)), lengh_control )  ]
    else:
        chocie = unique_pairs

    valid_pair = []
    for sup, que in chocie:
        support = sampling_res[sup]
        query = sampling_res[que]

        s_l = [x for y in support['label'] for x in y]
        q_l = [x for y in query['label'] for x in y]
        combed_type = list(set(s_l + q_l) )
        combed_type = [x for x in combed_type if x != 'O']
        test_string = {'support': support, 'query': query, 'types': combed_type }
        if set(s_l) == set(q_l) and len(combed_type) == 6:
            valid_pair.append(test_string)
        else:
            print('error')
            print(event)
            print(set(s_l + q_l))
    return valid_pair

res_final = []
for event in  train_events :
    print(event)
    sample = sampling_2_args[event]
    for key in tqdm( list(sample.keys()) ):
        sample = sampling_2_args[event]
        comb_list = sample[key]
        random.shuffle( comb_list )

        if len( sample[key] ) >= 2:
            if len(key) > 6:
                stored = []
                for _ in range( 2 ):
                    key1 = random.sample( key , 6)
                    if key1 not in stored:
                        stored.append(key1)
                        valid_pair = get_event_key_instance(new_train, event, key1, comb_list)
                        for pair in valid_pair:
                            res_final.append(pair)
            else:
                key1 = list(key)
                valid_pair = get_event_key_instance(new_train, event, key1, comb_list)
                for pair in valid_pair:
                    res_final.append(pair)
      
    print('total samples: ', len(res_final))

with jsonlines.open( prepath + split + '_6w2d.jsonl', 'w') as writer:
    writer.write_all(res_final)

# python sampling_6w2d.py