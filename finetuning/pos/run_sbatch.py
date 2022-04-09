#!/usr/bin/env python

import os

cmd_lines = {}

for lr in [3e-05, 5e-05, 1e-04]:
    for bsz in [16, 32, 64]:
        for max_epochs in [5, 6]:
            cmd_lines['tavbert_base_he_pos'] = f'''srun python run_ner_char_based.py --model_name_or_path \
            /checkpoints/roberta_base_lambda_5_oscar_3e-04/pytorch/ --task_name pos --preprocessing_num_workers 1 \
            --train_file finetuning/pos/data/raw_data_he/pos_train_multitag_chars_ud.json \
            --validation_file finetuning/pos/data/raw_data_he/pos_dev_multitag_chars_ud.json \
            --test_file finetuning/pos/data/raw_data_he/pos_test_multitag_chars_ud.json \
            --output_dir ./tavbert_base_he_pos_lr_{lr}_bsz_{bsz}_{max_epochs}epochs --do_train --do_eval \
            --do_predict --evaluation_strategy epoch --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
            --gradient_accumulation_steps {bsz // 8} --eval_accumulation_steps {bsz // 8} --learning_rate {lr} \
            --num_train_epochs {max_epochs} --warmup_ratio 0.1 --logging_dir ./tavbert_base_he_pos_lr_{lr}_bsz_{bsz}_ \
            {max_epochs}epochs --logging_strategy steps --logging_steps 10 --save_strategy epoch --fp16 --seed 1000 \
            --dataloader_num_workers 2 --dataloader_pin_memory --run_name tavbert_base_he_pos_lr_{lr}_bsz_{bsz}_ \
            {max_epochs}epochs --overwrite_output_dir 
             '''

            cmd_lines['alephbert_base_he_pos_all'] = f'''srun python run_ner.py \
            --model_name_or_path onlplab/alephbert-base --tokenizer_name \
            onlplab/alephbert-base --task_name pos --preprocessing_num_workers 1 \
            --train_file finetuning/pos/data/raw_data_he/pos_train_multitag_words_ud.json \
            --validation_file finetuning/pos/data/raw_data_he/pos_dev_multitag_words_ud.json \
            --test_file finetuning/pos/data/raw_data_he/pos_test_multitag_words_ud.json \
            --do_train --do_eval --do_predict --output_dir ./alephbert_base_he_pos_all_labels_lr_{lr}_bsz_{bsz}_ \
            {max_epochs}epochs --evaluation_strategy epoch \
            --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps {bsz // 8} \
            --eval_accumulation_steps {bsz // 8} --learning_rate {lr} --num_train_epochs {max_epochs} --warmup_ratio 0.1 \
            --logging_dir ./alephbert_base_he_pos_all_labels_lr_{lr}_bsz_{bsz}_{max_epochs}epochs --logging_strategy \
            steps --logging_steps 10 --save_strategy epoch --fp16 --seed 1000 --dataloader_num_workers 2 \
            --dataloader_pin_memory --run_name alephbert_base_he_pos_all_labels_lr_{lr}_bsz_{bsz}_{max_epochs}epochs \
            --overwrite_output_dir --label_all_tokens 
            '''

            for run_name, cmd_line in cmd_lines.items():
                job_file = os.path.join('.', f"{run_name}_lr_{lr}_bsz_{bsz}_{max_epochs}.job")
                with open(job_file, 'w') as fh:
                    fh.writelines("#!/bin/bash\n")
                    fh.writelines(f"#SBATCH --job-name={run_name}_lr_{lr}_bsz_{bsz}_{max_epochs}\n")
                    fh.writelines(f"#SBATCH --output={run_name}_lr_{lr}_bsz_{bsz}_{max_epochs}.out\n")
                    fh.writelines(f"#SBATCH --error={run_name}_lr_{lr}_bsz_{bsz}_{max_epochs}.err\n")
                    fh.writelines("#SBATCH --partition=killable\n")
                    fh.writelines("#SBATCH --time=2500\n")
                    fh.writelines("#SBATCH --signal=USR1@120\n")
                    fh.writelines("#SBATCH --nodes=1\n")
                    fh.writelines("#SBATCH --ntasks=1\n")
                    fh.writelines("#SBATCH --mem=50000\n")
                    fh.writelines("#SBATCH --cpus-per-task=4\n")
                    fh.writelines("#SBATCH --gpus=1\n")
                    fh.writelines(f"{cmd_line}\n")
                os.system("sbatch %s" % job_file)
