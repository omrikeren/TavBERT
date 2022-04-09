fairseq-preprocess --only-source --trainpref data/oscar_he/oscar_all.txt.train \
  --validpref data/oscar_he/oscar_all.txt.valid --destdir data/oscar_he/ --srcdict vocabulary/oscar_he/dict.txt \
  --workers 60 --char-tokenize
