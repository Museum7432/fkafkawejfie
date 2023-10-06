from rank_bm25 import BM25Okapi
import pandas as pd
import os
import py_vncorenlp


df = pd.read_json("datasets/ise-dsc01-warmup.json", orient="index")



rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=os.path.abspath('vncorenlp'))

def get_doc(id):
    ob = df.loc[id]
    context = ob["context"]
    context_sents = rdrsegmenter.word_segment(context)
    query = ' '.join(rdrsegmenter.word_segment(ob["claim"]))

    sents_concat = [context_sents[i] + context_sents[i + 1] for i in range(0,len(context_sents) - 1)]
    bm25 = BM25Okapi([doc.split(" ") for doc in sents_concat])

    query_score = bm25.get_scores(query=query.split(" "))
    final_doc_scores = [0 for _ in range(len(context_sents))]

    for i in range(len(query_score)):
        final_doc_scores[i] += query_score[i]
        final_doc_scores[i + 1] += query_score[i]

    sorted_ids = sorted(range(len(context_sents)), key=lambda k: final_doc_scores[k], reverse=True)
    fin_len = 0
    mask = [0 for _ in range(len(sorted_ids))]
    for i in sorted_ids:
        if fin_len + len(context_sents[i].split()) + len(query.split()) > 200:
            break
        
        fin_len += len(context_sents[i].split())
        mask[i] = 1

    selected = []
    for i in range(len(context_sents)):
        if mask[i] == 1:
            selected.append(context_sents[i])

    
    evidence_id = -1
    if ob["verdict"] != "NEI":
        evidence_sc = bm25.get_scores(''.join(rdrsegmenter.word_segment(ob["evidence"])).split(" "))
        evidence_id = evidence_sc.argmax()

    return [id, selected, query, ob["verdict"], None if evidence_id == -1 else [evidence_id], ob["domain"]]



re = pd.DataFrame(columns=["id","context", "claim", "verdict","evidence", "domain" ])

for index, _ in df.iterrows():
    re.loc[index] = get_doc(index)


re.to_json("../datasets/final.jsonl", orient='records',lines=True)

