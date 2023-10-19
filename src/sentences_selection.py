# from transformers import AutoTokenizer
import numpy as np

from rank_bm25 import BM25Okapi
from scipy.stats import truncnorm

# tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

def clean_str(s):
    return ''.join(e.lower() if e.isalnum() else " " for e in s)

def distribute_score(score, arr_len, pos):
    myclip_a = 0
    myclip_b = 1
    my_mean = pos/arr_len
    my_std = 0.7 / arr_len
    
    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std

    x_range = np.linspace(0,1,arr_len)

    y = truncnorm.pdf(x_range, a, b, loc = my_mean, scale = my_std)

    return score*y/sum(y)
    

def select_sentences(sentences, claim, tokenizer, force_sentences=[]):
    query = clean_str(claim).split()
    bm25 = BM25Okapi([clean_str(doc).split() for doc in sentences])

    query_score = bm25.get_scores(query=query)

    final_doc_scores = sum([distribute_score(score,len(query_score), idx) for idx, score in enumerate(query_score)])
    sorted_ids = sorted(range(len(sentences)), key=lambda k: final_doc_scores[k], reverse=True)

    claim_tk_length = len(tokenizer(claim).input_ids[:-1])
    abstract_len = tokenizer.model_max_length - claim_tk_length - 1
    context_tk_length = [len(tokenizer(s).input_ids[:-1]) for s in sentences]

    fin_len = sum([context_tk_length[i] for i in force_sentences])
    mask = [1 if i in force_sentences else 0 for i in range(len(sorted_ids))]

    for i in sorted_ids:
        if mask[i]:
            continue
        if fin_len + context_tk_length[i] > abstract_len:
            break
        
        fin_len += context_tk_length[i]
        mask[i] = 1
    
    return [i for i in range(len(sentences)) if mask[i] == 1]


