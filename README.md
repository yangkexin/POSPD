# POSPD
POS-constrained Parallel Decoding for Non-autoregressive Generation. Paper link: https://aclanthology.org/2021.acl-long.467.pdf.  
### Requirements
### Datasets
We use four datasets to evaluate POSPD, which can be obtained as follows:
- XSUM https://github.com/EdinburghNLP/XSum
- ROCStories Corpus https://cs.rochester.edu/nlp/rocstories/ We randomly sample 90k/4k stories for training/validation,
and the remaining 4160 for testing. Please first get access to the origin dataset and then contact us for the splited version.
- SQuAD 1.1 https://rajpurkar.github.io/SQuAD-explorer/
- WMT14 (DE-EN) https://www.statmt.org/wmt14/translation-task.html
### POS Tagging
Setup
> cd fairseq9/  
> pip install --editable .  

Here we take training POS Tagging on wmt14 as an example. 

0. Getting POS tagging data for BPE words  
To tag the BPE words, we propose a simple but effective subword-level POS tagging method for our POS predictor. which can be found in process_pos_data.ipynb

1. Preprossing data for fairseq  
As POS predictor needs the form of training example as <source sentence, target sentence, target POS sequnence>, we process these data repectively. Then, we put them into the same folder.  
- For <source sentence, target sentence>
> fairseq-preprocess \
--source-lang src --target-lang tgt \
--trainpref data/wmt14_data/train --validpref data/wmt14_data/valid \
--testpref data/wmt14_data/test \
--destdir data-bin/wmt14_data/bpe \
--workers 40 --joined-dictionary  

- For <target POS sequnence>    
> fairseq-preprocess \
   --source-lang tgt --target-lang tgt --only-source \
   --trainpref data/wmt14_data/pos/train --validpref data/wmt14_data/pos/valid --testpref data/wmt14_data/pos/test\
   --destdir data-bin/wmt14_data/bpe/pos \
   --workers 40 --joined-dictionary  
  
Then, renaming the pos file in data/wmt14_data/pos/ with ''pos'' as prefix. For example, test.src-tgt.tgt.bin should be renamed as postest.src-tgt.tgt.bin, and dict.tgt.txt should be renamed as posdict.tgt.txt.  
Finally, make sure all the datafile are in sibling directory named data/wmt14_data/bpe (remeber to place the datafile in "pos/" into the upper directory ).

2. Trainining POS Tagging  
> python ../fairseq_cli/trainpos.py \
   data-bin/wmt14_data/bpe \
   --arch pos_transformer_v2 --share-decoder-input-output-embed \
   --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
   --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
   --dropout 0.3 --weight-decay 0.0001 \
   --criterion label_smoothed_pos_cross_entropy_v2 --label-smoothing 0.1 \
   --max-tokens 6000 \
    --save-dir checkpoints/wmt14_transformer_pos_N6_v2_shallow\
    --fp16\
    --clip-norm 0.1 \
    --warmup-init-lr 1e-07 \
    --task translation_self_pos_v2\
    --tensorboard-logdir logfile/wmt14_transformer_pos_N6_v2_shallow\
   --eval-bleu \
   --eval-bleu-args '{"beam": 5}' \
   --eval-bleu-detok moses \
   --eval-bleu-remove-bpe \
   --eval-bleu-print-samples\
   --skip-invalid-size-inputs-valid-test \
   --encoder-layers 12 \
   --decoder-layers 1  
   
3. Generating POS Tagging
> python ../fairseq_cli/generatepos.py\
  data-bin/wmt14_data/bpe \
   --path checkpoints/wmt14_transformer_pos_N6_v2_shallow/checkpoint_best.pt \
   --batch-size 100 \
   --lenpen 2 \
   --beam 5 --remove-bpe --results-path #fill with the user dir# \
   --task translation_self_pos_v2
   
4. Processing POS Tagging for CMLM and Disco
   

### Constraining NAG
  
