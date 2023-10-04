from model import MultiVerSModel
import argparse
import os
import tqdm
temp_path = os.path.join("multivers_temp")

from model import MultiVerSModel
from data import get_dataloader


args = argparse.Namespace(
    checkpoint_path='multivers/checkpoints/fever_sci.ckpt',
    input_file=os.path.join("DataSets/test.jsonl"), 
    corpus_file=os.path.join(temp_path, "corpus.jsonl"), 
    output_file=os.path.join(temp_path, "output.jsonl"), 
    batch_size=1, 
    device=0, 
    num_workers=4, 
    no_nei=False, 
    force_rationale=False, 
    debug=False,
    # label_threshold=0.5,
    label_weight=3.0,
    rationale_weight=15.0,
    frac_warmup=1,
    encoder_name="vinai/phobert-base-v2",
    num_labels=3,
    lr=5e-5,
)

dataloader = enumerate(get_dataloader(args))
model = MultiVerSModel(args)


predictions_all = []

for idx,batch in dataloader:
    print(batch)
    preds_batch = model.predict(batch, args.force_rationale)
predictions_all.extend(preds_batch)

print(predictions_all)