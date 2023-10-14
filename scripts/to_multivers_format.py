from rank_bm25 import BM25Okapi
import pandas as pd
import os
import underthesea
import argparse
from tqdm import tqdm

def sentence_selection(context_sents, claim, n_token_limit=2500):

    sents_concat = [context_sents[i] + context_sents[i + 1] for i in range(0,len(context_sents) - 1)]

    bm25 = BM25Okapi([doc.split(" ") for doc in sents_concat])

    query=claim.split(" ")

    query_score = bm25.get_scores(query=query)

    final_doc_scores = [0 for _ in range(len(context_sents))]

    for i in range(len(query_score)):
        final_doc_scores[i] += query_score[i]
        final_doc_scores[i + 1] += query_score[i]

    sorted_ids = sorted(range(len(context_sents)), key=lambda k: final_doc_scores[k], reverse=True)

    fin_len = 0
    mask = [0 for _ in range(len(sorted_ids))]
    for i in sorted_ids:
        if fin_len + len(context_sents[i].split()) + len(query) > n_token_limit:
            break
        
        fin_len += len(context_sents[i].split())
        mask[i] = 1
    
    
    return [context_sents[i] for i in range(len(context_sents)) if mask[i] == 1]

def main():
    parser = argparse.ArgumentParser(
        description="transform custom data into multivers format"
    )
    
    parser.add_argument('--for_trainning', action='store_true',
                    help='if data has evidence field')
    
    parser.add_argument("--input_file", type=str,help="path to input file")

    parser.add_argument("--claim_file", type=str,help="path to output claim file")

    parser.add_argument("--corpus_file", type=str,help="path to output corpus file")

    parser.add_argument("--n_token_limit", type=int,default=2500)

    args = parser.parse_args()

    if not args.input_file:
        parser.error("input file needed!")

    if not args.claim_file:
        parser.error("claim output path need!")

    if not args.corpus_file:
        parser.error("corpus output path need!")
    
    if args.for_trainning:
        parser.error("for_trainning not implemented")
    
    tqdm.pandas()

    data = pd.read_json(args.input_file, orient="index")
    data.reset_index(inplace=True)

    data["context"] = data["context"].progress_apply(lambda text: underthesea.sent_tokenize(text))

    # create corpus file
    corpus = pd.DataFrame()
    corpus["doc_id"] = data["index"]
    corpus["abstract"] = data.progress_apply(lambda r: sentence_selection(r["context"], r["claim"], n_token_limit=args.n_token_limit), axis=1)
    corpus["title"] = data["index"].apply(lambda d: "")
    
    corpus.to_json(args.corpus_file,orient='records', lines=True)

    # create claims file
    claims = pd.DataFrame()
    claims["id"] = data["index"]
    claims["claim"] = data["claim"]
    claims["doc_ids"] = data["index"].apply(lambda id: [id])

    claims.to_json(args.claim_file,orient='records', lines=True)

if __name__ == "__main__":
    main()
