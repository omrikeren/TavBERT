We evaluate all three models on the respective sections of
the [Universal Dependencies](https://universaldependencies.org/) (UD) corpus: he_htb (Hebrew), ar_padt (Arabic), and
tr_imst (Turkish). The directory contains both the original raw data, and the preprocessed version obtained after
mapping from word-level to character-level labels. The preprocessed data is available in the segments format, as well as
in the multitag format.

To fine-tune on POS tagging, simply run ```python run_sbatch.py``` from the current directory.