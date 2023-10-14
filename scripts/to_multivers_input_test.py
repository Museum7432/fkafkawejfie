from rank_bm25 import BM25Okapi
import pandas as pd
import os
import underthesea

output_path = os.path.join("datasets")

data = pd.read_json("datasets/ise-dsc01-public-test-offcial.json", orient="index")

data.reset_index(inplace=True)

data = data.sample(100)

def sentence_selection(context_sents, claim):

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
        if fin_len + len(context_sents[i].split()) + len(query) > 2500:
            break
        
        fin_len += len(context_sents[i].split())
        mask[i] = 1
    
    
    return [context_sents[i] for i in range(len(context_sents)) if mask[i] == 1]


data["context"] = data["context"].apply(lambda text: underthesea.sent_tokenize(text))

corpus = pd.DataFrame()

corpus["doc_id"] = data["index"]
corpus["abstract"] = data.apply(lambda r: sentence_selection(r["context"], r["claim"]), axis=1)
corpus["title"] = data["index"].apply(lambda d: "")


claims = pd.DataFrame()
claims["id"] = data["index"]
claims["claim"] = data["claim"]
claims["doc_ids"] = data["index"].apply(lambda id: [id])

corpus.to_json( os.path.join(output_path,"corpus.jsonl"),orient='records', lines=True)
claims.to_json(os.path.join(output_path, "claims.jsonl"),orient='records', lines=True)
