# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('sequence_tagging')
class SequenceTaggingCriterion(FairseqCriterion):

    def __init__(self, task, classification_head_name, train_all_classification_heads=False):
        super().__init__(task)
        self.classification_head_name = classification_head_name

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            nargs='+',
                            default='sentence_classification_head',
                            help='name of the classification head to use')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        loss = None
        ncorrect = 0
        total_sample_size = 0
        for classification_head_name in self.classification_head_name:
            assert (
                    hasattr(model, 'classification_heads')
                    and classification_head_name in model.classification_heads
            ), 'model must provide sentence classification head for --criterion=sequence_tagging'

            logits, _ = model(
                **sample['net_input'],
                features_only=True,
                classification_head_name=classification_head_name,
            )
            targets = sample["target"][classification_head_name].view(-1)
            # print("Targets size:", targets.size())
            sample_size = sample['ntokens'] - sample['target'][classification_head_name].size(
                0)  # number of tokens without eos

            logits = logits.view(-1, logits.size(-1))
            # print("Logits size:", logits.size())
            if loss is None:
                loss = F.nll_loss(
                    F.log_softmax(logits, dim=-1, dtype=torch.float32),
                    targets,
                    ignore_index=-1,
                    reduction='sum',
                )
            else:
                loss += F.nll_loss(
                    F.log_softmax(logits, dim=-1, dtype=torch.float32),
                    targets,
                    ignore_index=-1,
                    reduction='sum',
                )

            masked_preds = logits[targets != -1].argmax(dim=1)
            masked_targets = targets[targets != -1]
            ncorrect += utils.item((masked_preds == masked_targets).sum())
            total_sample_size += sample_size

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'][classification_head_name].size(0),
            'sample_size': total_sample_size,
            'ncorrect': ncorrect
        }

        return loss, total_sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        nsentences = utils.item(sum(log.get('nsentences', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy', 100.0 * ncorrect / sample_size, nsentences, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
