from model import MultiVerSModel
import argparse
import os
import tqdm
temp_path = os.path.join("multivers_temp")
from transformers import AutoModel, AutoTokenizer


from model import MultiVerSModel
from data import get_dataloader

from lightning.pytorch.callbacks import ModelSummary

import torch

args = argparse.Namespace(
    checkpoint_path='checkpoints/fever.ckpt',
    input_file=os.path.join("datasets/final.jsonl"),
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
    label_threshold=None
)

model = MultiVerSModel(args)

model.load_state_dict(torch.load("checkpoints/fever_state_dict.ckpt", map_location="cpu")["state_dict"])

print(model)





dataloader = enumerate(get_dataloader(args))
for idx,batch in dataloader:
    # print(batch)
    preds_batch = model.predict(batch, args.force_rationale)
    print(preds_batch)



