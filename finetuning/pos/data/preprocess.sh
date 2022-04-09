fairseq-preprocess --only-source --trainpref pos_train_segmented_ud.txt_labels --validpref pos_dev_segmented_ud.txt_labels --testpref pos_test_segmented_ud.txt_labels --destdir postagging/label/

fairseq-preprocess --char-tokenize --destdir postagging/input --only-source --srcdict data/oscar/vocabulary/dict.txt -trainpref pos_train_segmented_ud.txt_tokens --validpref pos_dev_segmented_ud.txt_tokens --testpref pos_test_segmented_ud.txt_tokens --workers 1
