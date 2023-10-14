import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser(
        description="translate sentences in claim and corpus files"
    )

    parser.add_argument("--claims_file", type=str,help="path to claims file")

    parser.add_argument("--corpus_file", type=str,help="path to corpus file")

    parser.add_argument("--output_dir", type=str,help="save translated chunks to this folder")

    parser.add_argument("--chunk_size", type=int,help="split into chunks to save checkpoint", default=250)


    args = parser.parse_args()

    # load vinai-translate-vi2en

    tokenizer = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en", src_lang="vi_VN")  
    model = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en")
    model = nn.DataParallel(model)

    
    # translate corpus
    corpus = pd.read_json(args.corpus_file,orient="records", lines = True)

    corpus_output_path = os.path.join(args.output_dir,"corpus")

    number_of_sections = corpus.shape[0] // args.chunk_size
    if number_of_sections <= 0:
        number_of_sections = 1

    for idx,df in enumerate(np.array_split(corpus, number_of_sections)):
        print(idx)
        chunk_output_path = os.path.join(corpus_output_path,"corpus_" + str(idx) + ".jsonl")

        if os.path.isfile(chunk_output_path):
            continue

        # split the abstract field into rows

        df = df.explode("abstract").reset_index(drop=True)

        texts = df["abstract"].to_list()

        input_ids = tokenizer(texts, padding=True, return_tensors="pt").to("cuda")

        output_ids = model.module.generate(
            **input_ids,
            decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"],
            num_return_sequences=1,
            num_beams=5,
            early_stopping=True
        )

        re = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        df["abstract"] = re

        df = df.groupby(["doc_id", "title"],sort=False)["abstract"].apply(list).reset_index(name='abstract')

        df.to_json(chunk_output_path,orient='records', lines=True)
    

    # translate claims

    claims = pd.read_json(args.claims_file,orient="records", lines = True)

    claims_output_path = os.path.join(args.output_dir,"claims")

    number_of_sections = claims.shape[0] // args.chunk_size
    if number_of_sections <= 0:
        number_of_sections = 1

    for idx,df in enumerate(np.array_split(corpus, number_of_sections)):
        print(idx)

        chunk_output_path = os.path.join(claims_output_path,"claims_" + str(idx) + ".jsonl")

        if os.path.isfile(chunk_output_path):
            continue
        
        texts = df["claim"].to_list()

        input_ids = tokenizer(texts, padding=True, return_tensors="pt").to("cuda")

        output_ids = model.module.generate(
            **input_ids,
            decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"],
            num_return_sequences=1,
            num_beams=5,
            early_stopping=True
        )

        re = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        df["claim"] = re

        df.to_json(chunk_output_path,orient='records', lines=True)

if __name__ == "__main__":
    main()