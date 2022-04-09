export MKL_SERVICE_FORCE_INTEL=1

fairseq-train --fp16 data/oscar_tr --task masked_lm --criterion masked_lm --arch roberta_base --tokens-per-sample 2048 \
  --max-tokens 8096 --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
  --lr-scheduler polynomial_decay --lr 3e-04 --total-num-update 125000 --dropout 0.1 --attention-dropout 0.1 \
  --weight-decay 0.01 --update-freq 32 --max-update 125000 --log-format simple --log-interval 1 \
  --skip-invalid-size-inputs-valid-test --poisson-lambda 5 --shorten-method truncate --sample-break-mode complete_doc \
  --num-workers 2 --warmup-updates 3000 --save-dir checkpoints/roberta_base_lambda_5_oscar_3e-04_tr \
  --validate-interval-updates 1000
