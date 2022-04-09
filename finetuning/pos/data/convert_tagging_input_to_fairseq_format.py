import argparse
import os

sentence_id_prefixes = ["mst", "assabah", "afp", "alhayat", "annahar", "ummah", "xinhua"]


def main(input_dir):
    txt_input_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    for input_file in txt_input_files:
        labels = []
        tokens = []
        file_txt_list = []
        tokens_file_txt_list = []

        with open(os.path.join(input_dir, input_file), 'r', encoding='utf-8') as fobj:
            for line in fobj:
                line = line.strip()
                if any([line.startswith(prefix) for prefix in sentence_id_prefixes]):
                    continue
                try:
                    sent_id = int(line)
                except ValueError:
                    if line == '':
                        file_txt_list.append(' '.join(labels))
                        tokens_file_txt_list.append(''.join(tokens))
                        labels = []
                        tokens = []
                        continue
                    if line == 'VOID':
                        label = 'VOID'
                        token = ' '
                    elif 'morph' in input_file and line == 'X':
                        label = 'VOID'
                        token = ' '
                    else:
                        token, label = line.split()
                    labels.append(label)
                    tokens.append(token)
        open(os.path.join(input_dir, "{}_labels".format(input_file)), 'w', encoding='utf-8').write(
            '\n\n'.join(file_txt_list))
        open(os.path.join(input_dir, "{}_tokens".format(input_file)), 'w', encoding='utf-8').write(
            '\n\n'.join(tokens_file_txt_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str)
    args = parser.parse_args()
    main(args.input_dir)
