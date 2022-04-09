import argparse
import re
from operator import itemgetter

p = re.compile("^.*INFO | valid | epoch (?P<epoch>[0-9]+) | valid on 'valid' subset | loss (?P<loss>[0-9]\.[0-9]+).*$")


def find_best_k_checkpoints(log_file, k=5):
    epochs = {}
    with open(log_file, 'r') as f:
        for line in f:
            m = p.match(line)
            if m is not None:
                print(m.groupdict())
                ep = int(m.groupdict()['epoch'])
                loss = float(m.groupdict()['loss'])
                print("Found epoch {}: {:.3f}".format(ep, loss))
                epochs[ep] = loss

    return dict(sorted(epochs.items(), key=itemgetter(1))[:k])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, help="the path to the log file to parse.")
    parser.add_argument("--k", type=int, help="return the best k checkpoints.")
    args = parser.parse_args()
    print(find_best_k_checkpoints(args.log_file, args.k))
