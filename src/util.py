import numpy as np
from rank_bm25 import BM25Okapi
from scipy.stats import truncnorm
import re
import numpy as np
import os
from underthesea import sent_tokenize
import math
import copy


# def remove_noises(text):
#     text = text.replace(".\n\n", ". ")
#     text = text.replace(".\n", ". ")

#     text = text.replace(":\n\n", ": ")
#     text = text.replace("?\n\n", "?. ")

#     text = text.replace("\n\n", ". ")
#     text = text.replace("\n", ". ")
#     return text

def remove_noises(text):
    text = text.replace(".\n\n", ". ")
    text = text.replace(".\n", ". ")

    text = text.replace("\n\n*", ".* ")

    text = text.replace(":\n\n", ": ")
    # text = text.replace("?\n\n", "? ")

    text = text.replace("?\n\n", ", ")
    text = text.replace("?", ", ")

    text = text.replace("\n\n", ". ")

    # incorrect but somehow has a higher accuracy on trainning data
    # text = text.replace("\n\n", ", ")

    text = text.replace("\n", ", ")
    return text

def no_accent_vietnamese(s):
    s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
    s = re.sub(r'[ÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪ]', 'A', s)
    s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
    s = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', s)
    s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
    s = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s)
    s = re.sub(r'[ÌÍỊỈĨ]', 'I', s)
    s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
    s = re.sub(r'[ƯỪỨỰỬỮÙÚỤỦŨ]', 'U', s)
    s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
    s = re.sub(r'[ỲÝỴỶỸ]', 'Y', s)
    s = re.sub(r'[Đ]', 'D', s)
    s = re.sub(r'[đ]', 'd', s)
    return s


# def sentences_splitting(text, rdrsegmenter, word_segmented=False):

#     text_lower = no_accent_vietnamese(text.lower())
    
#     original_sents=[]
#     processed_sents = []
    
#     text_pos = 0
#     for _, sent in rdrsegmenter.annotate_text(text).items():
#         words = [w["wordForm"] for w in sent]

#         if word_segmented:
#             processed_sents.append(" ".join(words))
#             words = [w.replace("_", " ") if w != "_" else "_" for w in words]
#         else:
#             processed_sents.append(" ".join(words))
#             words = [w.replace("_", " ") if w != "_" else "_" for w in words]

#         processed_sents.append(" ".join(words))

#         words = [no_accent_vietnamese(w.lower()) for w in words]
#         words = sum([w.split() for w in words],[])


#         # find position of words in text_lower
#         pos_start = []
#         pos_end = []

#         for w in words:
#             idx = text_lower.find(w,text_pos)

#             pos_start.append(idx)
#             pos_end.append(idx + len(w))

#             text_pos = idx + len(w)
        
#         # for testing
#         # test = [text_lower[pos_start[i]: pos_end[i]] for i in range(len(pos_start))]
#         # assert test == words

#         original_sents.append(text[pos_start[0]: pos_end[-1]])
#     return original_sents, processed_sents


def sentences_splitting(text, word_segmented=False, rdrsegmenter=None):

    original_sents = sent_tokenize(text)

    original_sents = [s for sent in original_sents for s in sent.split("... ")]
    original_sents = [s for sent in original_sents for s in sent.split(".. ")]
    original_sents = [s for sent in original_sents for s in sent.split(". ")]
    original_sents = [s.strip() for s in original_sents]

    if word_segmented:
        processed_sents = [" ".join(rdrsegmenter.word_segment(s)) for s in original_sents]
    else:
        processed_sents = None
    
    return original_sents, processed_sents

def get_longest_continuest_context(tokenized_sents_lenght, start_pos, max_token_number):
    if start_pos >= len(tokenized_sents_lenght):
        return [], 0
    selected = []
    curr_length = 0
    for idx in range(start_pos, len(tokenized_sents_lenght)):
        if curr_length + tokenized_sents_lenght[idx] > max_token_number:
            break
        selected.append(idx)
        curr_length += tokenized_sents_lenght[idx]
    
    return selected, curr_length

def get_best_continuest_context(tokenized_sents_lenght, query_score, max_token_number=200):
    best_context = []
    max_score = 0
    for i in range(len(tokenized_sents_lenght)):
        selected, curr_length = get_longest_continuest_context(tokenized_sents_lenght, start_pos=i, max_token_number=max_token_number)
        score = sum([query_score[i] for i in selected])
        if score > max_score:
            best_context = selected
            max_score = score
    return best_context, max_score

def slice_context(tokenized_sents_lenght, number_of_contexts, query_score, max_token_number=200):
    qscores = copy.copy(query_score)
    selected = []

    for _ in range(number_of_contexts*2):
        if len(selected) >= number_of_contexts:
            break
        best_context, max_score = get_best_continuest_context(
            tokenized_sents_lenght=tokenized_sents_lenght,
            query_score=qscores,
            max_token_number=max_token_number
        )
        
        # reduce the score of the selected context

        for i in best_context:
            qscores[i] /= 3
        # print(max_score)
        if best_context not in selected:
            selected.append(best_context)
    return selected


def clean_str(s):
    return ''.join(e.lower() if e.isalnum() else " " for e in no_accent_vietnamese(s))

def distribute_score(score, arr_len, pos):
    if arr_len == 1:
        return (arr_len - 1)
    myclip_a = 0
    myclip_b = 1
    my_mean = pos/arr_len
    my_std = 0.75 / arr_len
    
    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std

    x_range = np.linspace(0,1,arr_len)

    y = truncnorm.pdf(x_range, a, b, loc = my_mean, scale = my_std)

    return score*y/sum(y)

def get_bm25_scores(sents, query):
    query = clean_str(query).split()

    bm25 = BM25Okapi([clean_str(doc).split() for doc in sents])

    query_score = bm25.get_scores(query=query)

    return query_score

def get_top_until_filled(scores, tokenized_sents_lenght, max_token_number):
    sorted_ids = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)

    fin_len = 0
    mask = [0 for _ in range(len(sorted_ids))]

    for i in sorted_ids:
        if mask[i]:
            continue
        if fin_len + tokenized_sents_lenght[i] > max_token_number:
            break
        
        fin_len += tokenized_sents_lenght[i]
        mask[i] = 1
    
    return [i for i in range(len(scores)) if mask[i] == 1]

def context_slicing(sents, claim, tokenizer, silce_size = 500, max_size=4000, max_number_of_slices=None):
    
    query_score = get_bm25_scores(sents, claim)

    normalized_scores = sum([distribute_score(score,len(query_score), idx) for idx, score in enumerate(query_score)])
    
    # claim_tk_length = len(tokenizer(claim).input_ids[:-1])

    context_tk_length = [len(tokenizer(s).input_ids[:-1]) for s in sents]

    # abstract_len = model_max_length - claim_tk_length - 1

    most_relevance = get_top_until_filled(
        scores = query_score,
        tokenized_sents_lenght=context_tk_length,
        max_token_number=silce_size
    )

    full = most_relevance = get_top_until_filled(
        scores = query_score,
        tokenized_sents_lenght=context_tk_length,
        max_token_number=max_size
    )

    if max_number_of_slices and max_number_of_slices <= 1:
        return [most_relevance]

    number_of_contexts = math.ceil(sum(context_tk_length)*1.25/silce_size)

    if max_number_of_slices:
        number_of_contexts = min(max_number_of_slices, number_of_contexts)

    slices = slice_context(
        tokenized_sents_lenght = context_tk_length,
        number_of_contexts=number_of_contexts, 
        query_score=normalized_scores, 
        max_token_number=silce_size
    )


    if most_relevance not in slices:
        slices.append(most_relevance)
    
    return slices, full