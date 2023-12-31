from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import lightning_getattr
import math
import torch
from torch import nn
from torch.nn import functional as F

from transformers import LongformerModel, RobertaModel, AutoTokenizer
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention

import os

from allennlp_nn_util import batched_index_select
from allennlp_feedforward import FeedForward



def unbatch(d, ignore=[]):
    """
    Convert a dict of batched tensors to a list of tensors per entry. Ignore any
    keys in the list.
    """
    ignore = set(ignore)

    to_unbatch = {}
    for k, v in d.items():
        # Skip ignored keys.
        if k in ignore:
            continue
        if isinstance(v, torch.Tensor):
            # Detach and convert tensors to CPU.
            new_v = v.detach().cpu().numpy()
        else:
            new_v = v

        to_unbatch[k] = new_v

    # Make sure all entries have same length.
    lengths = [len(v) for v in to_unbatch.values()]
    if len(set(lengths)) != 1:
        raise ValueError("All values must be of same length.")

    res = []
    for i in range(lengths[0]):
        to_append = {}
        for k, v in to_unbatch.items():
            to_append[k] = v[i]

        res.append(to_append)

    return res



class RobertaLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value = None,
        output_attentions=False,
    ):
        attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = any(is_index_global_attn.flatten())
        return super().forward(hidden_states, 
                               is_index_masked=is_index_masked, 
                               is_index_global_attn=is_index_global_attn, 
                               is_global_attn=is_global_attn,
                               attention_mask=attention_mask, 
                               output_attentions=output_attentions)

class RobertaLongModel(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = RobertaLongSelfAttention(config, layer_id=i)



class MultiVerSModel(pl.LightningModule):
    """
    Multi-task SciFact model that encodes claim / abstract pairs using
    Longformer and then predicts rationales and labels in a multi-task fashion.
    """
    def __init__(self, hparams):
        """
        Arguments are set by `add_model_specific_args`.
        """
        super().__init__()
        self.save_hyperparameters()

        # Constants
        self.nei_label = 1  # Int category for NEI label.

        # Classificaiton thresholds. These were added later, so older configs
        # won't have them.
        if hasattr(hparams, "label_threshold"):
            self.label_threshold = hparams.label_threshold
        else:
            self.label_threshold = None

        if hasattr(hparams, "rationale_threshold"):
            self.rationale_threshold = hparams.rationale_threshold
        else:
            self.rationale_threshold = 0.5

        # Paramters
        self.label_weight = hparams.label_weight
        self.rationale_weight = hparams.rationale_weight
        self.frac_warmup = hparams.frac_warmup

        # Model components.
        self.encoder_name = hparams.encoder_name
        self.encoder = self._get_encoder(hparams)
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)

        # Final output layers.
        hidden_size = self.encoder.config.hidden_size
        activations = [nn.GELU(), nn.Identity()]
        dropouts = [self.dropout.p, 0]
        self.label_classifier = FeedForward(
            input_dim=2 * hidden_size,
            num_layers=2,
            hidden_dims=[hidden_size, hparams.num_labels],
            activations=activations,
            dropout=dropouts)
        self.rationale_classifier = FeedForward(
            input_dim=2 * hidden_size,
            num_layers=2,
            hidden_dims=[hidden_size, 1],
            activations=activations,
            dropout=dropouts)

        # Learning rates.
        self.lr = hparams.lr

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        encoder: The transformer encoder that gets the embeddings.
        label_weight: The weight to assign to label prediction in the loss function.
        rationale_weight: The weight to assign to rationale selection in the loss function.
        num_labels: The number of label categories.
        gradient_checkpointing: Whether to use gradient checkpointing with Longformer.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--encoder_name", type=str, default="bluenguyen/longformer-phobert-base-4096")
        parser.add_argument("--label_weight", type=float, default=1.0)
        parser.add_argument("--rationale_weight", type=float, default=15.0)
        parser.add_argument("--num_labels", type=int, default=3)
        parser.add_argument("--gradient_checkpointing", action="store_true")
        parser.add_argument("--lr", type=float, default=5e-5)
        parser.add_argument("--frac_warmup", type=float, default=0.1,
                            help="The fraction of training to use for warmup.")
        parser.add_argument("--scheduler_total_epochs", default=None, type=int,
                            help="If given, pass as total # epochs to LR scheduler.")
        parser.add_argument("--label_threshold", default=None, type=float,
                            help="Threshold for non-NEI label.")
        parser.add_argument("--rationale_threshold", default=0.5, type=float,
                            help="Threshold for rationale.")

        return parser

    @staticmethod
    def _get_encoder(hparams):
        "Start from the Longformer science checkpoint."

        def get_phobert_longformer():
            if not os.path.isdir("checkpoints/phobert_4096"):
                os.makedirs(os.path.dirname("./checkpoints"), exist_ok=True)
                model = RobertaLongModel.from_pretrained("bluenguyen/longformer-phobert-base-4096")
                model.save_pretrained("checkpoints/phobert_4096")

        get_phobert_longformer()
        
        starting_encoder_name = "bluenguyen/longformer-phobert-base-4096"
        encoder = LongformerModel.from_pretrained(
            "checkpoints/phobert_4096",
            gradient_checkpointing=hparams.gradient_checkpointing
        )

        # layers_to_train = [
        #     "pooler.dense",
        #     "encoder.layer"
        # ]
        
        # for name, param in encoder.named_parameters():
        #     if name.startswith(tuple(layers_to_train)):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        return encoder

    def forward(self, tokenized, abstract_sent_idx):
        """
        Run the forward pass. Encode the inputs and return softmax values for
        the labels and the rationale sentences.

        The `abstract_sent_idx` gives the indices of the `</s>` tokens being
        used to represent each sentence in the abstract.
        """
        # Encode.
        encoded = self.encoder(**tokenized)

        # Make label predictions.
        pooled_output = self.dropout(encoded.pooler_output)
        
        # Make rationale predictions
        # Need to invoke `continguous` or `batched_index_select` can fail.
        hidden_states = self.dropout(encoded.last_hidden_state).contiguous()
        sentence_states = batched_index_select(hidden_states, abstract_sent_idx)

        # Concatenate the CLS token with the sentence states.
        pooled_rep = pooled_output.unsqueeze(1).expand_as(sentence_states)
        # [n_documents x max_n_sentences x (2 * encoder_hidden_dim)]
        rationale_input = torch.cat([pooled_rep, sentence_states], dim=2)
        # Squeeze out dim 2 (the encoder dim).
        # [n_documents x max_n_sentences]
        rationale_logits = self.rationale_classifier(rationale_input).squeeze(2)

        # Predict rationales.
        # [n_documents x max_n_sentences]
        rationale_probs = torch.sigmoid(rationale_logits.detach())
        predicted_rationales = (rationale_probs >= self.rationale_threshold).to(torch.int64)

        # sentences' relavance scores
        # [n_documents x max_n_sentences]
        relavance_scores = F.softmax(rationale_logits, dim=-1)
        # attention over sentence_states
        sentence_att = torch.matmul(relavance_scores.unsqueeze(1), sentence_states).squeeze(1)
        
        # perform label predictions
        label_input = torch.cat([sentence_att, pooled_output], dim=-1)
        # [n_documents x n_labels]
        label_logits = self.label_classifier(label_input)

        # Predict labels.
        # [n_documents]

        label_probs = F.softmax(label_logits.detach(), dim=1)
        if self.label_threshold is None:
            # If not doing a label threshold, just take the largest.
            predicted_labels = label_logits.argmax(dim=1)
        else:
            # If we're using a threshold, set the score for the null label to
            # the threshold and take the largest.
            label_probs_truncated = label_probs.clone()
            label_probs_truncated[:, self.nei_label] = self.label_threshold
            predicted_labels = label_probs_truncated.argmax(dim=1)

        return {"label_logits": label_logits,
                "rationale_logits": rationale_logits,
                "label_probs": label_probs,
                "rationale_probs": rationale_probs,
                "predicted_labels": predicted_labels,
                "predicted_rationales": predicted_rationales}

    def predict(self, batch, force_rationale=False):
        """
        Make predictions on a batch passed in from the dataloader.
        """
        # Run forward pass.
        with torch.no_grad():
            output = self(batch["tokenized"], batch["abstract_sent_idx"])

        return self.decode(output, batch, force_rationale)

    @staticmethod
    def decode(output, batch, force_rationale=False):
        """
        Run decoding to get output in human-readable form. The `output` here is
        the output of the forward pass.
        """
        # Mapping from ints to labels.
        label_lookup = {0: "CONTRADICT",
                        1: "NEI",
                        2: "SUPPORT"}

        # Get predicted rationales, only keeping eligible sentences.
        instances = unbatch(batch, ignore=["tokenized"])
        output_unbatched = unbatch(output)

        predictions = []
        for this_instance, this_output in zip(instances, output_unbatched):
            predicted_label = label_lookup[this_output["predicted_labels"]]

            # Due to minibatching, may need to get rid of padding sentences.
            rationale_ix = this_instance["abstract_sent_idx"] > 0
            rationale_indicators = this_output["predicted_rationales"][rationale_ix]
            predicted_rationale = rationale_indicators.nonzero()[0].tolist()
            # Need to convert from numpy data type to native python.
            predicted_rationale = [int(x) for x in predicted_rationale]

            # If we're forcing a rationale, then if the predicted label is not "NEI"
            # take the highest-scoring sentence as a rationale.
            if predicted_label != "NEI" and not predicted_rationale and force_rationale:
                candidates = this_output["rationale_probs"][rationale_ix]
                predicted_rationale = [candidates.argmax()]

            res = {
                "claim_id": int(this_instance["claim_id"]),
                "abstract_id": int(this_instance["abstract_id"]),
                "predicted_label": predicted_label,
                "predicted_rationale": predicted_rationale,
                "label_probs": this_output["label_probs"],
                "rationale_probs": this_output["rationale_probs"][rationale_ix]
            }
            predictions.append(res)

        return predictions



class model_wrapper:
    def __init__(self, checkpoint_path, device ="cuda", tokenizer_name="bluenguyen/longformer-phobert-base-4096", strict=True):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        model = MultiVerSModel.load_from_checkpoint(checkpoint_path=checkpoint_path, map_location=device, strict=strict)

        self.device = device

        model.to(device)
        model.eval()
        model.freeze()
        del model.hparams["hparams"].precision  # Don' use 16-bit precision during evaluation.

        self.model = model
    
    def _tensorizes_one(self, claim_id, doc_id ,sentences, claim):
        def _get_global_attention_mask(tokenized):
            "Assign global attention to all special tokens and to the claim."
            input_ids = torch.tensor(tokenized.input_ids)
            # Get all the special tokens.
            is_special = (input_ids == self.tokenizer.bos_token_id) | (
                input_ids == self.tokenizer.eos_token_id
            )
            # Get all the claim tokens (everything before the first </s>).
            first_eos = torch.where(input_ids == self.tokenizer.eos_token_id)[0][0]
            is_claim = torch.arange(len(input_ids)) < first_eos
            # Use global attention if special token, or part of claim.
            global_attention_mask = is_special | is_claim
            # Unsqueeze to put in batch form, and cast like the tokenizer attention mask.
            global_attention_mask = global_attention_mask.to(torch.int64)
            return global_attention_mask.tolist()
        def _get_abstract_sent_tokens(tokenized, title):
            "Get the indices of the </s> tokens representing each abstract sentence."
            is_eos = torch.tensor(tokenized["input_ids"]) == self.tokenizer.eos_token_id
            eos_idx = torch.where(is_eos)[0]
            # If there's a title, the first two </s> tokens are for the claim /
            # abstract separator and the title. Keep the rest.
            # If no title, keep all but the first.
            start_ix = 1 if title is None else 2
            return eos_idx[start_ix:].tolist()

        cited_text = self.tokenizer.eos_token.join(sentences)
        tokenized = self.tokenizer(claim + self.tokenizer.eos_token + cited_text)
        tokenized["global_attention_mask"] = _get_global_attention_mask(tokenized)

        abstract_sent_idx = _get_abstract_sent_tokens(tokenized, None)
        assert len(abstract_sent_idx) == len(sentences)
        return {
                "claim_id": claim_id,
                "abstract_id": doc_id,
                "tokenized": tokenized,
                "abstract_sent_idx": abstract_sent_idx,
            }
        
    def _tensorizes_batch(self ,sents_list, claim_list, idx_list):
        def to_tensor(batch, field):
            return torch.tensor([x[field] for x in batch])

        def _pad(xxs, pad_value):
            """
            Pad a list of lists to the length of the longest entry, using the given
            `pad_value`.
            """
            res = []
            max_length = max(map(len, xxs))
            for entry in xxs:
                to_append = [pad_value] * (max_length - len(entry))
                padded = entry + to_append
                res.append(padded)

            return torch.tensor(res)

        def _pad_field(entries, field_name, pad_value):
            xxs = [entry[field_name] for entry in entries]
            return _pad(xxs, pad_value)

        def _pad_tokenized(tokenized):
            """
            Pad the tokenizer outputs. Need to do this manually because the
            tokenizer's default padder doesn't expect `global_attention_mask` as an
            input.
            """
            fields = ["input_ids", "attention_mask", "global_attention_mask"]
            pad_values = [self.tokenizer.pad_token_id, 0, 0]
            tokenized_padded = {}
            for field, pad_value in zip(fields, pad_values):
                tokenized_padded[field] = _pad_field(tokenized, field, pad_value)

            return tokenized_padded

        batch = [self._tensorizes_one(idx, idx, se, cl) for se, cl, idx in zip(sents_list, claim_list, idx_list)]


        res = {
            "claim_id": to_tensor(batch, "claim_id"),
            "abstract_id": to_tensor(batch, "abstract_id"),
            "tokenized":_pad_tokenized([x["tokenized"] for x in batch]),
            "abstract_sent_idx": _pad_field(batch, "abstract_sent_idx", 0),
        }
        assert res.keys() == batch[0].keys()
        return res
    
    def to_device(self, batch):
        for k in batch["tokenized"].keys():
            batch["tokenized"][k] = batch["tokenized"][k].to(self.device)

        batch["abstract_sent_idx"] = batch["abstract_sent_idx"].to(self.device)
        
    def predict(self, batch):
        sents_list = [i["sentences"] for i in batch]
        claim_list = [i["claim"] for i in batch]
        idx_list = [i for i in range(len(batch))]

        tokenized = self._tensorizes_batch(
            sents_list=sents_list,
            claim_list=claim_list,
            idx_list=idx_list
        )

        self.to_device(tokenized)

        outputs = self.model.predict(tokenized)

        return outputs
