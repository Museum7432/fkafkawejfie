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

from util import sentences_splitting, remove_noises, context_slicing, get_bm25_scores


def split_sents(r, word_segmented=False, rdrsegmenter=None):
        original_sents, processed_sents = sentences_splitting(r["context"], word_segmented=word_segmented, rdrsegmenter=rdrsegmenter)
        return pd.Series([original_sents, processed_sents])

def get_evidence_index(sents, evidence):
        if not evidence or len(evidence) == 0:
            return []
        scores = get_bm25_scores(
            sents=sents,
            query=evidence
        )
        return [scores.argmax()]

def slice_context_wrapper(r, tokenizer, max_number_of_slices = None):
    sents = r["sents"]
    claim = r["claim"]

    full = [i for i in range(len(sents))]

    if max_number_of_slices == 1:
        return [full]

    slices = context_slicing(
        sents=sents,
        claim=claim,
        tokenizer=tokenizer,
        silce_size=500,
        max_number_of_slices= None if not max_number_of_slices else max_number_of_slices-1
    )

    if full not in slices:
        slices.append(full)
    
    return slices

def get_actual_verdict_evidence(r):
    old_verdict = r["verdict"]
    if old_verdict == "NEI":
        return pd.Series(["NEI", []])
    
    evidence_id = r["evidence"][0]

    if evidence_id in r["context_ids"]:
        return pd.Series([old_verdict, [r["context_ids"].index(evidence_id)]])
    else:
        return pd.Series(["NEI", []])

def to_multivers_lable(label):
    if label == "SUPPORTED":
        return "SUPPORT"
    if label == "REFUTED":
        return "CONTRADICT"

    raise ValueError('Unknown label: ' + str(label))

def get_evidence_dict(r):
    if r["verdict_n"] == "NEI":
        return {}
    
    re = {}

    re[r["doc_id"]] = [{
        "label": to_multivers_lable(r["verdict_n"]),
        "sentences":r["evidence_n"]
    }]

    return re

def concat_evi(evis):
    re = dict()
    for evi in evis:
        re.update(evi)
    return re

def to_train_label(label):
    if label in ["NEI", "NOT ENOUGH INFO"]:
        return "NOT ENOUGH INFO"

    if label == "SUPPORTED":
        return "SUPPORTS"

    if label == "REFUTED":
        return "REFUTES"
    raise ValueError("unknown label: " + str(label))

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

    parser.add_argument("--word_segmented", action='store_true',help="apply vncorenlp's word segmentation")


    parser.add_argument("--use_alternative_format", action='store_true',help="for big corpus")
    parser.add_argument("--output_file", type=str,help="path to output file (for alternative format)")

    parser.add_argument("--sample_size", type=int,default=0)

    parser.add_argument("--max_number_of_slices", type=int,default=None)

    # parser.add_argument("--n_token_limit", type=int,default=2500)

    args = parser.parse_args()

    if not args.input_file:
        parser.error("input file needed!")

    # if not args.claims_file:
    #     parser.error("claim output path needed!")

    # if not args.corpus_file:
    #     parser.error("corpus output path needed!")
    
    tqdm.pandas()

    temp = os.getcwd()
    if args.word_segmented:
        rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=abspath('vncorenlp'))
    else:
        rdrsegmenter = None

    os.chdir(temp)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)


    data = pd.read_json(abspath(args.input_file), orient="index")
    data.reset_index(inplace=True)

    if args.sample_size > 0:
        data = data.sample(args.sample_size)

    data["context"] = data["context"].progress_apply(remove_noises)

    if args.word_segmented:
        data["claim"] = data["claim"].progress_apply(lambda c: ' '.join(rdrsegmenter.word_segment(c)))

    data[["sents", "processed_sents"]] = data.progress_apply(lambda r:split_sents(r,word_segmented=args.word_segmented, rdrsegmenter=rdrsegmenter), axis=1)

    data.drop("context", axis=1,inplace=True)

    if args.for_trainning:
        data["evidence"] = data.progress_apply(lambda r: get_evidence_index(r["sents"], r["evidence"]), axis=1)


    data["context_ids"] = data.progress_apply(lambda r: slice_context_wrapper(r, tokenizer=tokenizer, max_number_of_slices=args.max_number_of_slices), axis=1)

    t1 = data.explode("context_ids").reset_index().rename(columns={'index': 'id'})

    if args.for_trainning:
        t1[["verdict_n", "evidence_n"]] = t1.progress_apply(get_actual_verdict_evidence, axis=1)


    if args.use_alternative_format:
        re = pd.DataFrame()
        re["abstract_id"] = t1.index
        re["id"] = t1["id"]
        re["claim"] = t1["claim"]
        re["label"] = t1["verdict_n"].apply(to_train_label)
        # re["label"] = t1["verdict_n"]
        if args.word_segmented:
            re["sentences"] = t1.apply(lambda r: [r["processed_sents"][i] for i in r["context_ids"]], axis=1)
        else:
            re["sentences"] = t1.apply(lambda r: [r["sents"][i] for i in r["context_ids"]], axis=1)
        re["evidence_sets"] = t1["evidence_n"].apply(lambda e: [] if len(e) == 0 else [e])
        re["negative_sample_id"] = 0

        re = re.sample(frac=1).reset_index(drop=True)
        
        re.to_json(args.output_file, orient='records', lines=True, force_ascii=False)

        print("done!")

        return
    # create corpus file

    corpus = pd.DataFrame()
    corpus["doc_id"] = t1.index
    if args.word_segmented:
        corpus["abstract"] = t1.apply(lambda r: [r["processed_sents"][i] for i in r["context_ids"]], axis=1)
        corpus["original"] = t1.apply(lambda r: [r["sents"][i] for i in r["context_ids"]], axis=1)
    else:
        corpus["abstract"] = t1.apply(lambda r: [r["sents"][i] for i in r["context_ids"]], axis=1)

    corpus["title"] = None
    corpus.to_json(abspath(args.corpus_file),orient='records', lines=True, force_ascii=False)

    # create claims file

    t2 = t1.reset_index().rename(columns={'index': 'doc_id'})

    claims = pd.DataFrame()
    claims.index = t2.index
    claims["id"] = t2["id"]

    if args.for_trainning:
        claims["cited_doc_ids"] = t2["doc_id"]
    else:
        claims["doc_ids"] = t2["doc_id"]
    claims["claim"] = t2["claim"]

    if args.for_trainning:
        claims["verdict_n"] = t2["verdict_n"]
        claims["evidence_n"] = t2["evidence_n"]

        claims["evidence"] = t2.progress_apply(get_evidence_dict, axis=1)
        claims.drop(["verdict_n", "evidence_n"],axis=1, inplace=True)

    claims = claims.groupby(["id", "claim"], sort=False).agg(list).reset_index()

    if args.for_trainning:
        claims["evidence"] = claims["evidence"].progress_apply(concat_evi)

    claims.to_json(abspath(args.claims_file),orient='records', lines=True, force_ascii=False)

    print("done!")

if __name__ == "__main__":
    main()
