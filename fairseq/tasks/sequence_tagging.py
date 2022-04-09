# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np

from fairseq.data import (
    data_utils,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    OffsetTokensDataset,
    RightPadDataset,
    SortDataset,
    ReplaceDataset,
)
from fairseq.data.assert_same_length_dataset import AssertSameLengthDataset
from fairseq.tasks import register_task, LegacyFairseqTask

logger = logging.getLogger(__name__)


@register_task('sequence_tagging')
class SequenceTaggingTask(LegacyFairseqTask):
    """
    Sequence tagging (also called sentence tagging or sequence labelling) task that predicts a class for each input token.
    Inputs should be stored in 'input' directory, labels in 'label' directory.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', nargs='+', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--num-classes', nargs='+', type=int, default=-1,
                            help='number of classes')
        parser.add_argument('--no-shuffle', action='store_true', default=False)

    def __init__(self, args, data_dictionary, label_dictionary):
        super().__init__(args)
        self.dictionary = data_dictionary
        self._label_dictionary = label_dictionary
        if not hasattr(args, 'max_positions'):
            self._max_positions = (
                args.max_source_positions,
                args.max_target_positions,
            )
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        cls.mask_idx = dictionary.add_symbol('<mask>')
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_dict = {}
        label_dict = {}
        if isinstance(args.classification_head_name, list):
            classification_heads_info = zip(args.classification_head_name, args.num_classes, [args.data])
        else:
            classification_heads_info = [(args.classification_head_name, args.num_classes, args.data)]
        for classification_head_name, num_classes, data in classification_heads_info:
            assert num_classes > 0, 'Must set --num-classes'

            # load data dictionary
            data_dict = cls.load_dictionary(
                args,
                os.path.join(data, 'input', 'dict.txt'),
                source=True,
            )
            logger.info('[input] dictionary: {} types'.format(len(data_dict)))

            # load label dictionary
            label_dict[classification_head_name] = cls.load_dictionary(
                args,
                os.path.join(data, 'label', 'dict.txt'),
                source=False,
            )
            logger.info('[label] dictionary: {} types'.format(len(label_dict[classification_head_name])))

        return SequenceTaggingTask(args, data_dict, label_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        def get_path(type, data, split):
            return os.path.join(data, type, split)

        def make_dataset(type, data, dictionary):
            split_path = get_path(type, data, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            assert dataset is not None, 'could not find dataset: {}'.format(get_path(type, split))
            return dataset

        dataset = None
        for classification_head_name, data in zip(self.args.classification_head_name, self.args.data):
            src_tokens = make_dataset('input', data, self.source_dictionary)

            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_tokens))

            label_dataset = make_dataset('label', data, self.label_dictionary[classification_head_name])

            if dataset is None:
                dataset = {
                    'id': IdDataset(),
                    'net_input': {
                        'src_tokens': RightPadDataset(
                            src_tokens,
                            pad_idx=self.source_dictionary.pad(),
                        ),
                        'src_lengths': NumelDataset(src_tokens, reduce=False),
                    },
                    'target': {},
                    'nsentences': NumSamplesDataset(),
                    'ntokens': NumelDataset(src_tokens, reduce=True),
                    '_assert_lengths_match': {},
                }

            dataset['target'][classification_head_name] = RightPadDataset(
                # use -1 as padding, will be used to mask out padding when calculating loss
                ReplaceDataset(
                    # replace eos and existing padding (used when some tokens should not be predicted) with -1
                    OffsetTokensDataset(  # offset tokens to get the targets to the correct range (0,1,2,...)
                        label_dataset,
                        offset=-self.label_dictionary[classification_head_name].nspecial,
                    ),
                    replace_map={
                        self.label_dictionary[classification_head_name].eos() - self.label_dictionary[
                            classification_head_name].nspecial: -1,
                        self.label_dictionary[classification_head_name].pad() - self.label_dictionary[
                            classification_head_name].nspecial: -1,
                    },
                    offsets=np.zeros(len(label_dataset), dtype=np.int)
                ),
                pad_idx=-1
            )

            dataset['_assert_lengths_match'][classification_head_name] = AssertSameLengthDataset(src_tokens,
                                                                                                 label_dataset)

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
        )

        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)

        if isinstance(args.classification_head_name, list):
            classification_heads_info = zip(args.classification_head_name, args.num_classes)
        else:
            classification_heads_info = [(args.classification_head_name, args.num_classes)]

        for classification_head_name, num_classes in classification_heads_info:
            model.register_classification_head(
                classification_head_name,
                num_classes=num_classes,
                sequence_tagging=True
            )

        return model

    def max_positions(self):
        return self._max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary
