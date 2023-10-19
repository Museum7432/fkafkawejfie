import os, sys
sys.path.append("src")

from os.path import abspath
from rank_bm25 import BM25Okapi
import pandas as pd

import argparse
from tqdm import tqdm
from sentences_selection import select_sentences
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

def truncate_abstracts(claim, sentences, tokenizer, evidence_sets=[]):

    selected = select_sentences(
        sentences=sentences,
        claim=claim,
        tokenizer=tokenizer,
        force_sentences=evidence_sets
    )

    new_evidence_sets = [selected.index(old_id) for old_id in evidence_sets]

    return [sentences[i] for i in selected], new_evidence_sets

def process_row(row, tokenizer=tokenizer):
    evidence_sets = row["evidence_sets"]
    if len(evidence_sets) != 0:
        evidence_sets = evidence_sets[0]

    sents, new_evidence_sets = truncate_abstracts(

        claim=row["claim"],
        sentences=row["sentences"],
        tokenizer=tokenizer,
        evidence_sets=evidence_sets,
    )
    if len(new_evidence_sets)!= 0:
        new_evidence_sets = [new_evidence_sets]
    
    return pd.Series([sents, new_evidence_sets])

def main():
    parser = argparse.ArgumentParser(
        description="truncate abstract of dataset"
    )
    
    parser.add_argument('--for_trainning', action='store_true',
                    help='if data has evidence field')
    
    parser.add_argument("--input_file", type=str,help="path to input file")

    parser.add_argument("--output_file", type=str,help="path to output file")

    parser.add_argument("--tokenizer", type=str,help="name of tokenizer (for truncating the context)",default="xlm-roberta-large")


    args = parser.parse_args()

    if not args.input_file:
        parser.error("input file needed!")

    if not args.output_file:
        parser.error("output file needed!")

    
    tqdm.pandas()

    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    data = pd.read_json(abspath(args.input_file), orient="records", lines=True)

    data[['sentences',"evidence_sets"]] = data.progress_apply(process_row,axis=1)

    data.to_json(args.output_file, orient='records', lines=True)


if __name__ == "__main__":
    main()
