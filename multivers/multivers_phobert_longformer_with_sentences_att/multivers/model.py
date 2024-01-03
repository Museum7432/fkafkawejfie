from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import lightning_getattr
import math
import torch
from torch.utils.checkpoint import checkpoint   # Without this, I get an import error.
from torch import nn
from torch.nn import functional as F
import transformers
from transformers.optimization import get_linear_schedule_with_warmup
from pytorch_lightning.core.decorators import auto_move_data

from transformers import LongformerModel, RobertaModel
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention

from allennlp_nn_util import batched_index_select
from allennlp_feedforward import FeedForward
from metrics import SciFactMetrics
import os
import util


def masked_binary_cross_entropy_with_logits(input, target, weight, rationale_mask):
    """
    Binary cross entropy loss. Ignore values where the target is -1. Compute
    loss as a "mean of means", first taking the mean over the sentences in each
    row, and then over all the rows.
    """
    # Mask to indicate which values contribute to loss.
    mask = torch.where(target > -1, 1, 0)

    # Need to convert target to float, and set -1 values to 0 in order for the
    # computation to make sense. We'll ignore the -1 values later.
    float_target = target.clone().to(torch.float)
    float_target[float_target == -1] = 0
    losses = F.binary_cross_entropy_with_logits(
        input, float_target, reduction="none")
    # Mask out the values that don't matter.
    losses = losses * mask

    # Take "sum of means" over the sentence-level losses for each instance.
    # Take means so that long documents don't dominate.
    # Multiply by `rationale_mask` to ignore sentences where we don't have
    # rationale annotations.
    n_sents = mask.sum(dim=1)

    n_sents = torch.max(n_sents, torch.ones_like(n_sents))

    totals = losses.sum(dim=1)
    means = totals / n_sents
    final_loss = (means * weight * rationale_mask).sum()

    return final_loss

def get_mask_for_correct_binary_prediction(pred, target, lower_threshold=-0.1, upper_threshold=0.1):
    with torch.no_grad():
        lower_mask = torch.logical_and(
            pred <= lower_threshold,
            torch.where(target == 0, 1, 0)
        )

        upper_mask = torch.logical_and(
            pred >= upper_threshold,
            torch.where(target == 1, 1, 0)
        )

        return torch.logical_or(lower_mask, upper_mask)

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

class MultiVerSModel_learnable_weight(pl.LightningModule):
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
            input_dim=hidden_size*2,
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

        self.sigmoid_to_softmax_W = nn.Parameter(torch.tensor(25.0), requires_grad=True)

        # Learning rates.
        self.lr = hparams.lr

        # Metrics
        fold_names = ["train", "valid", "test"]
        metrics = {}
        for name in fold_names:
            metrics[f"metrics_{name}"] = SciFactMetrics(compute_on_step=False)

        self.metrics = nn.ModuleDict(metrics)

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
        starting_encoder_name = "bluenguyen/longformer-phobert-base-4096"

        def get_phobert_longformer():
            if not os.path.isdir("checkpoints/phobert_4096"):
                os.makedirs(os.path.dirname("./checkpoints"), exist_ok=True)
                model = RobertaLongModel.from_pretrained("bluenguyen/longformer-phobert-base-4096")
                model.save_pretrained("checkpoints/phobert_4096")

        get_phobert_longformer()
        
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
        # convert sigmoid logits to softmax
        relavance_scores = F.softmax(rationale_logits * self.sigmoid_to_softmax_W, dim=-1)

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

    def training_step(self, batch, batch_idx):
        "Multi-task loss on a batch of inputs."
        res = self(batch["tokenized"], batch["abstract_sent_idx"])

        # Loss for label prediction.
        label_loss = F.cross_entropy(
            res["label_logits"], batch["label"], reduction="none")
        # Take weighted average of per-sample losses.
        label_loss = (batch["weight"] * label_loss).sum()

        # mask = get_mask_for_correct_binary_prediction(
        #     res["rationale_logits"],
        #     batch["rationale"],
        #     lower_threshold=-0.1,
        #     upper_threshold=0.1
        # )
        #
        # tmp = res["rationale_logits"].detach() * (~mask)
        #
        # masked_rationale_logits = res["rationale_logits"] * mask + tmp
        
        # Loss for rationale selection.
        rationale_loss = masked_binary_cross_entropy_with_logits(
            res["rationale_logits"], batch["rationale"], batch["weight"],
            batch["rationale_mask"])

        # Loss is a weighted sum of the two components.
        loss = self.label_weight * label_loss + self.rationale_weight * rationale_loss

        # Invoke metrics.
        self.log("label_loss", label_loss.detach())
        self.log("rationale_loss", rationale_loss.detach())
        self.log("loss", loss.detach())

        self.log("sigmoid_to_softmax_W", self.sigmoid_to_softmax_W.detach())

        self._invoke_metrics(res, batch, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch["tokenized"], batch["abstract_sent_idx"])
        self._invoke_metrics(pred, batch, "valid")

    def validation_epoch_end(self, outs):
        "Log metrics at end of validation."
        # Log the train metrics here so that we keep track of train and valid
        # metrics at the same time, even if we validate multiple times an epoch.
        self._log_metrics("train")
        self._log_metrics("valid")

    def test_step(self, batch, batch_idx):
        pred = self(batch["tokenized"], batch["abstract_sent_idx"])
        self._invoke_metrics(pred, batch, "test")

    def test_epoch_end(self, outs):
        "Log metrics at end of test."
        # As above, log the train metrics together with the test metrics.
        self._log_metrics("train")
        self._log_metrics("test")

    def _invoke_metrics(self, pred, batch, fold):
        """
        Invoke metrics for a single step of train / validation / test.
        `batch` is gold, `pred` is prediction, `fold` specifies the fold.
        """
        assert fold in ["train", "valid", "test"]

        # We won't need gradients.
        detached = {k: v.detach() for k, v in pred.items()}
        # Invoke the metrics appropriate for this fold.
        self.metrics[f"metrics_{fold}"](detached, batch)

    def _log_metrics(self, fold):
        "Log metrics for this epoch."
        the_metric = self.metrics[f"metrics_{fold}"]
        to_log = the_metric.compute()
        the_metric.reset()
        for k, v in to_log.items():
            self.log(f"{fold}_{k}", v)

        # Uncomment this if still hanging.
            # self.log(f"{fold}_{k}", v, sync_dist=True, sync_dist_op="sum")

    def configure_optimizers(self):
        "Set the same LR for all parameters."
        hparams = self.hparams.hparams
        optimizer = transformers.AdamW(self.parameters(), lr=self.lr)

        # If we're debugging, just use the vanilla optimizer.
        if hparams.fast_dev_run or hparams.debug:
            return optimizer

        # Calculate total number of training steps, for the optimizer.
        if isinstance(hparams.gpus, str):
            # If gpus is a string, count the number by splitting on commas.
            n_gpus = len([x for x in hparams.gpus.split(",") if x])
        else:
            n_gpus = int(hparams.gpus)

        steps_per_epoch = math.ceil(
            hparams.num_training_instances /
            (n_gpus * hparams.train_batch_size * hparams.accumulate_grad_batches))

        if hparams.scheduler_total_epochs is not None:
            n_epochs = hparams.scheduler_total_epochs
        else:
            n_epochs = hparams.max_epochs

        num_steps = n_epochs * steps_per_epoch
        warmup_steps = num_steps * self.frac_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps)

        lr_dict = {"scheduler": scheduler,
                   "interval": "step"}
        res = {"optimizer": optimizer,
               "lr_scheduler": lr_dict}

        return res

    @auto_move_data
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
        instances = util.unbatch(batch, ignore=["tokenized"])
        output_unbatched = util.unbatch(output)

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


class MultiVerSModel_seperate_output(pl.LightningModule):
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
            input_dim=hidden_size*2,
            num_layers=2,
            hidden_dims=[hidden_size, hparams.num_labels],
            activations=activations,
            dropout=dropouts)

        # first value is for softmax, second for sigmoid
        self.rationale_classifier = FeedForward(
            input_dim=2 * hidden_size,
            num_layers=2,
            hidden_dims=[hidden_size, 2],
            activations=activations,
            dropout=dropouts)

        # Learning rates.
        self.lr = hparams.lr

        # Metrics
        fold_names = ["train", "valid", "test"]
        metrics = {}
        for name in fold_names:
            metrics[f"metrics_{name}"] = SciFactMetrics(compute_on_step=False)

        self.metrics = nn.ModuleDict(metrics)

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
        parser.add_argument("--rationale_weight", type=float, default=30.0)
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
        starting_encoder_name = "bluenguyen/longformer-phobert-base-4096"

        def get_phobert_longformer():
            if not os.path.isdir("checkpoints/phobert_4096"):
                os.makedirs(os.path.dirname("./checkpoints"), exist_ok=True)
                model = RobertaLongModel.from_pretrained("bluenguyen/longformer-phobert-base-4096")
                model.save_pretrained("checkpoints/phobert_4096")

        get_phobert_longformer()
        
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
        # rationale_logits = self.rationale_classifier(rationale_input).squeeze(2)

        rationale_logits_raw = self.rationale_classifier(rationale_input)

        # softmax_rationale_logits = rationale_logits_raw[:, :, 0]
        # sigmoid_rationale_logits = rationale_logits_raw[:, :, 1]

        rationale_logits = rationale_logits_raw[:, :, 1]

        # Predict rationales.
        # [n_documents x max_n_sentences]
        rationale_probs = torch.sigmoid(rationale_logits.detach())
        predicted_rationales = (rationale_probs >= self.rationale_threshold).to(torch.int64)

        # sentences' relavance scores
        # [n_documents x max_n_sentences]
        relavance_scores = F.softmax(rationale_logits_raw[:, :, 0], dim=-1)

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

    def training_step(self, batch, batch_idx):
        "Multi-task loss on a batch of inputs."
        res = self(batch["tokenized"], batch["abstract_sent_idx"])

        # Loss for label prediction.
        label_loss = F.cross_entropy(
            res["label_logits"], batch["label"], reduction="none")
        # Take weighted average of per-sample losses.
        label_loss = (batch["weight"] * label_loss).sum()


        mask = get_mask_for_correct_binary_prediction(
            res["rationale_logits"],
            batch["rationale"],
            lower_threshold=-1,
            upper_threshold=1
        )

        # tmp = res["rationale_logits"].detach() * (~mask)
        # masked_rationale_logits = res["rationale_logits"] * mask

        rationale = batch["rationale"]
        rationale[mask] = -1

        # Loss for rationale selection.
        rationale_loss = masked_binary_cross_entropy_with_logits(
            res["rationale_logits"], rationale, batch["weight"],
            batch["rationale_mask"])

        # Loss is a weighted sum of the two components.
        loss = self.label_weight * label_loss + self.rationale_weight * rationale_loss

        # Invoke metrics.
        self.log("label_loss", label_loss.detach())
        self.log("rationale_loss", rationale_loss.detach())
        self.log("loss", loss.detach())

        self._invoke_metrics(res, batch, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch["tokenized"], batch["abstract_sent_idx"])
        self._invoke_metrics(pred, batch, "valid")

    def validation_epoch_end(self, outs):
        "Log metrics at end of validation."
        # Log the train metrics here so that we keep track of train and valid
        # metrics at the same time, even if we validate multiple times an epoch.
        self._log_metrics("train")
        self._log_metrics("valid")

    def test_step(self, batch, batch_idx):
        pred = self(batch["tokenized"], batch["abstract_sent_idx"])
        self._invoke_metrics(pred, batch, "test")

    def test_epoch_end(self, outs):
        "Log metrics at end of test."
        # As above, log the train metrics together with the test metrics.
        self._log_metrics("train")
        self._log_metrics("test")

    def _invoke_metrics(self, pred, batch, fold):
        """
        Invoke metrics for a single step of train / validation / test.
        `batch` is gold, `pred` is prediction, `fold` specifies the fold.
        """
        assert fold in ["train", "valid", "test"]

        # We won't need gradients.
        detached = {k: v.detach() for k, v in pred.items()}
        # Invoke the metrics appropriate for this fold.
        self.metrics[f"metrics_{fold}"](detached, batch)

    def _log_metrics(self, fold):
        "Log metrics for this epoch."
        the_metric = self.metrics[f"metrics_{fold}"]
        to_log = the_metric.compute()
        the_metric.reset()
        for k, v in to_log.items():
            self.log(f"{fold}_{k}", v)

        # Uncomment this if still hanging.
            # self.log(f"{fold}_{k}", v, sync_dist=True, sync_dist_op="sum")

    def configure_optimizers(self):
        "Set the same LR for all parameters."
        hparams = self.hparams.hparams
        optimizer = transformers.AdamW(self.parameters(), lr=self.lr)

        # If we're debugging, just use the vanilla optimizer.
        if hparams.fast_dev_run or hparams.debug:
            return optimizer

        # Calculate total number of training steps, for the optimizer.
        if isinstance(hparams.gpus, str):
            # If gpus is a string, count the number by splitting on commas.
            n_gpus = len([x for x in hparams.gpus.split(",") if x])
        else:
            n_gpus = int(hparams.gpus)

        steps_per_epoch = math.ceil(
            hparams.num_training_instances /
            (n_gpus * hparams.train_batch_size * hparams.accumulate_grad_batches))

        if hparams.scheduler_total_epochs is not None:
            n_epochs = hparams.scheduler_total_epochs
        else:
            n_epochs = hparams.max_epochs

        num_steps = n_epochs * steps_per_epoch
        warmup_steps = num_steps * self.frac_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps)

        lr_dict = {"scheduler": scheduler,
                   "interval": "step"}
        res = {"optimizer": optimizer,
               "lr_scheduler": lr_dict}

        return res

    @auto_move_data
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
        instances = util.unbatch(batch, ignore=["tokenized"])
        output_unbatched = util.unbatch(output)

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


