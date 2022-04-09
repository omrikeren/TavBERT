# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re

from tokenizers import BertWordPieceTokenizer

SPACE_NORMALIZER = re.compile(r"\s+")

worpiece_tokenizer = None


def get_wordpiece_tokenizer(vocab):
    global worpiece_tokenizer
    if worpiece_tokenizer is None:
        worpiece_tokenizer = BertWordPieceTokenizer(vocab, unk_token='<unk>',
                                                    sep_token='</s>',
                                                    cls_token='<s>',
                                                    pad_token='<pad>',
                                                    mask_token='<mask>')
    return worpiece_tokenizer


def tokenize_line(line, *args, **kwargs):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def tokenize_line_by_wordpiece(line, vocab):
    tokenizer = get_wordpiece_tokenizer(vocab)
    return tokenizer.encode(line, add_special_tokens=False).tokens
