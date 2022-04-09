#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import json
import sys
from collections import Counter
from multiprocessing import Pool


class CharacterEncoder:

    def __init__(self, encoder, errors='replace'):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.cache = {}

        try:
            import regex as re
            self.re = re
        except ImportError:
            raise ImportError('Please install regex with: pip install regex')

    def encode(self, text):
        tokens = []
        for ch in text:
            tokens.append(self.encoder.get(ch, 1000))
        return tokens

    def decode(self, tokens):
        return ''.join([self.decoder[token] for token in tokens])


def main():
    """
    Helper script to encode raw text with the character tokenizer using multiple processes.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-json",
        help='path to encoder.json',
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=['-'],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=['-'],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    assert len(args.inputs) == len(args.outputs), \
        "number of input and output paths should match"

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-" else sys.stdin
            for input in args.inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-" else sys.stdout
            for output in args.outputs
        ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)

        stats = Counter()
        for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, outputs):
                    print(enc_line, file=output_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)


class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        global char_enc
        char_enc = get_encoder(self.args.encoder_json)

    def encode(self, line):
        global char_enc
        ids = char_enc.encode(line)
        return list(map(str, ids))

    def decode(self, tokens):
        global char_enc
        return char_enc.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]
            tokens = self.encode(line)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]


def get_encoder(encoder_json_path):
    with open(encoder_json_path, 'r') as f:
        encoder = json.load(f)
    return CharacterEncoder(
        encoder=encoder
    )


if __name__ == "__main__":
    main()
