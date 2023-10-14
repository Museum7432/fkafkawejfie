import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

tqdm.pandas()

tokenizer = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en", src_lang="vi_VN")  
model = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en")
model.cuda()


corpus = pd.read_json("datasets/corpus.jsonl",orient="records", lines = True)


def translate(texts):
    input_ids = tokenizer(texts, padding=True, return_tensors="pt").to("cuda")

    output_ids = model.generate(
        **input_ids,
        decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)

corpus["abstract"] = corpus["abstract"].progress_apply(lambda abstract: translate(abstract))

corpus.to_json( "datasets/corpus_translated.jsonl",orient='records', lines=True)


claims = pd.read_json("datasets/claims.jsonl",orient="records", lines = True)

claims["claim"] = claims["claim"].progress_apply(lambda claim: translate([claim])[0])

claims.to_json( "datasets/claims_translated.jsonl",orient='records', lines=True)