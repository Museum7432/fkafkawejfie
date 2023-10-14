from rank_bm25 import BM25Okapi
import pandas as pd
import os
import py_vncorenlp
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=os.path.abspath('vncorenlp'))

for_trainning = True
output_path = os.path.join("/home/arch/Projects/college/dsc/datasets")

data = pd.read_json("/home/arch/Projects/college/dsc/datasets/ise-dsc01-warmup.json", orient="index")
data.reset_index(inplace=True)

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
        if fin_len + len(context_sents[i].split()) + len(query) > 200:
            break
        
        fin_len += len(context_sents[i].split())
        mask[i] = 1
    
    selected = []
    for i in range(len(context_sents)):
        if mask[i] == 1:
            selected.append(context_sents[i])
    
    return selected

def get_evidence(doc_id, context_sents, evidence, verdict):
    if verdict == "NEI":
        return None
    bm25 = BM25Okapi([sent.split(" ") for sent in context_sents])

    evidence_sc = bm25.get_scores(evidence.split(" "))

    evidence_id = evidence_sc.argmax()

    re = {}

    re[doc_id] = [{
        "sentences":[evidence_id],
        "label":verdict
    }]
    return re



data["claim"] = data["claim"].apply(lambda c: ' '.join(rdrsegmenter.word_segment(c)))
data["evidence"] = data["evidence"].apply(lambda e: ' '.join(rdrsegmenter.word_segment(e) if e else ""))

data["context"] = data["context"].apply(lambda text: rdrsegmenter.word_segment(text))


corpus = pd.DataFrame()

corpus["doc_id"] = data["index"]
corpus["abstract"] = data.apply(lambda r: sentence_selection(r["context"], r["claim"]), axis=1)
corpus["title"] = data["index"].apply(lambda d: "")


claims = pd.DataFrame()
claims["id"] = data["index"]
claims["claim"] = data["claim"]
claims["doc_ids"] = data["index"].apply(lambda id: [id])

if for_trainning:
    claims["evidence"] = data.apply(lambda r: get_evidence(r["index"], r["context"], r["evidence"], r["verdict"]), axis=1)


corpus.to_json( os.path.join(output_path,"corpus.jsonl"),orient='records', lines=True)
claims.to_json(os.path.join(output_path, "claims.jsonl"),orient='records', lines=True)
