import os, sys
sys.path.append("src")

from os.path import abspath
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
import pandas as pd
# import underthesea
import py_vncorenlp
import argparse
from tqdm import tqdm
import re
from sentences_selection import select_sentences

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

def sentences_splitting(text, rdrsegmenter):

    text_lower = no_accent_vietnamese(text.lower())
    
    original_sents=[]
    processed_sents = []
    
    text_pos = 0
    for _, sent in rdrsegmenter.annotate_text(text).items():
        words = [w["wordForm"] for w in sent]
        words = [w.replace("_", " ") if w != "_" else "_" for w in words]

        processed_sents.append(" ".join(words))

        words = [no_accent_vietnamese(w.lower()) for w in words]
        words = sum([w.split() for w in words],[])


        # find position of words in text_lower
        pos_start = []
        pos_end = []

        for w in words:
            idx = text_lower.find(w,text_pos)

            pos_start.append(idx)
            pos_end.append(idx + len(w))

            text_pos = idx + len(w)
        
        # for testing
        # test = [text_lower[pos_start[i]: pos_end[i]] for i in range(len(pos_start))]
        # assert test == words

        original_sents.append(text[pos_start[0]: pos_end[-1]])
    return original_sents, processed_sents

def context_slicing(context,claim, tokenizer, rdrsegmenter):
    original_sents, processed_sents = sentences_splitting(
        text=context,
        rdrsegmenter=rdrsegmenter
    )

    selected = select_sentences(
        sentences=processed_sents,
        claim=claim,
        tokenizer=tokenizer
    )

    return [original_sents[i] for i in selected]

def main():
    parser = argparse.ArgumentParser(
        description="transform custom data into multivers format"
    )
    
    parser.add_argument('--for_trainning', action='store_true',
                    help='if data has evidence field')
    
    parser.add_argument("--input_file", type=str,help="path to input file")

    parser.add_argument("--claims_file", type=str,help="path to output claims file")

    parser.add_argument("--corpus_file", type=str,help="path to output corpus file")

    parser.add_argument("--tokenizer", type=str,help="name of tokenizer (for truncating the context)",default="xlm-roberta-large")

    # parser.add_argument("--n_token_limit", type=int,default=2500)

    args = parser.parse_args()

    if not args.input_file:
        parser.error("input file needed!")

    if not args.claims_file:
        parser.error("claim output path needed!")

    if not args.corpus_file:
        parser.error("corpus output path needed!")
    
    if args.for_trainning:
        parser.error("for_trainning not implemented")
    
    tqdm.pandas()

    temp = os.getcwd()
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=abspath('vncorenlp'))
    os.chdir(temp)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)


    data = pd.read_json(abspath(args.input_file), orient="index")
    data.reset_index(inplace=True)

    data["context"] = data["context"].progress_apply(lambda text: text.replace(".\n\n",". ").replace(".\n\n",". ").replace("\n\n",". ").replace("\n",". "))
    # data["context"] = data["context"].progress_apply(lambda text: underthesea.sent_tokenize(text))

    # create corpus file
    corpus = pd.DataFrame()
    corpus["doc_id"] = data["index"]
    
    print("extracting abstracts...")
    corpus["abstract"] = data.progress_apply(lambda r: context_slicing(r["context"], r["claim"], tokenizer=tokenizer, rdrsegmenter=rdrsegmenter), axis=1)
    corpus["title"] = data["index"].apply(lambda d: None)
    
    corpus.to_json(abspath(args.corpus_file),orient='records', lines=True, force_ascii=False)

    # create claims file
    claims = pd.DataFrame()
    claims["id"] = data["index"]
    claims["claim"] = data["claim"]
    claims["doc_ids"] = data["index"].apply(lambda id: [id])

    claims.to_json(abspath(args.claims_file),orient='records', lines=True, force_ascii=False)

if __name__ == "__main__":
    main()
