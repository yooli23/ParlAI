set -xue

# middle size model (2.7B) + data augmentation, raw blender, no hacks
# parlai train_model -t inspired_blender_baseline \
#     -veps 0.25 \
#     --attention-dropout 0.0 \
#     --batchsize 1 \
#     --model transformer/generator \
#     --embedding-size 2560 \
#     --ffn-size 10240 \
#     --variant prelayernorm \
#     --n-heads 32 \
#     --n-positions 128 \
#     --n-encoder-layers 2 \
#     --n-decoder-layers 24 \
#     --history-add-global-end-token end \
#     --delimiter '  ' \
#     --dict-tokenizer bytelevelbpe  \
#     --dropout 0.1 \
#     --fp16 True \
#     --init-model zoo:blender/blender_3B/model \
#     --dict-file zoo:blender/blender_3B/model.dict \
#     --label-truncate 128 \
#     --log_every_n_secs 10 \
#     -lr 7e-06 \
#     --lr-scheduler reduceonplateau \
#     --lr-scheduler-patience 3 \
#     --optimizer adafactor \
#     --relu-dropout 0.0 \
#     --activation gelu \
#     --model-parallel true \
#     --save-after-valid True \
#     --text-truncate 128 \
#     --truncate 128 \
#     --warmup_updates 100 \
#     --fp16-impl mem_efficient \
#     --update-freq 2 \
#     --gradient-clip 0.1 \
#     --skip-generation True \
#     -vp 10 \
#     -vmt ppl \
#     -vmm min \
#     --model-file ./experiments/inspired_blender_baseline/model/blender_1019_exp1 \
#     -eps 3.0

# parlai train_model -t inspired_blender_baseline \
#     -veps 0.25 \
#     --attention-dropout 0.0 \
#     --batchsize 1 \
#     --model transformer/generator \
#     --embedding-size 2560 \
#     --ffn-size 10240 \
#     --variant prelayernorm \
#     --n-heads 32 \
#     --n-positions 128 \
#     --n-encoder-layers 2 \
#     --n-decoder-layers 24 \
#     --history-add-global-end-token end \
#     --delimiter '  ' \
#     --dict-tokenizer bytelevelbpe  \
#     --dropout 0.1 \
#     --fp16 True \
#     --init-model zoo:blender/blender_3B/model \
#     --dict-file zoo:blender/blender_3B/model.dict \
#     --label-truncate 128 \
#     --log_every_n_secs 10 \
#     -lr 7e-06 \
#     --lr-scheduler reduceonplateau \
#     --lr-scheduler-patience 3 \
#     --optimizer adafactor \
#     --relu-dropout 0.0 \
#     --activation gelu \
#     --model-parallel true \
#     --save-after-valid True \
#     --text-truncate 128 \
#     --truncate 128 \
#     --warmup_updates 100 \
#     --fp16-impl mem_efficient \
#     --update-freq 2 \
#     --gradient-clip 0.1 \
#     --skip-generation True \
#     -vp 10 \
#     -vmt ppl \
#     -vmm min \
#     --model-file ./experiments/inspired_blender_baseline/model/blender_1028_exp2 \
#     -eps 3.0

parlai train_model -t inspired_blender_baseline \
    -veps 0.25 \
    --attention-dropout 0.0 \
    --batchsize 2 \
    --model transformer/generator \
    --embedding-size 2560 \
    --ffn-size 10240 \
    --variant prelayernorm \
    --n-heads 32 \
    --n-positions 128 \
    --n-encoder-layers 2 \
    --n-decoder-layers 24 \
    --history-add-global-end-token end \
    --delimiter '  ' \
    --dict-tokenizer bytelevelbpe  \
    --dropout 0.1 \
    --fp16 True \
    --init-model zoo:blender/blender_3B/model \
    --dict-file zoo:blender/blender_3B/model.dict \
    --label-truncate 128 \
    --log_every_n_secs 10 \
    -lr 7e-06 \
    --lr-scheduler reduceonplateau \
    --lr-scheduler-patience 3 \
    --optimizer adafactor \
    --relu-dropout 0.0 \
    --activation gelu \
    --model-parallel true \
    --save-after-valid True \
    --text-truncate 128 \
    --truncate 128 \
    --warmup_updates 100 \
    --fp16-impl mem_efficient \
    --update-freq 2 \
    --gradient-clip 0.1 \
    --skip-generation True \
    -vp 10 \
    -vmt ppl \
    -vmm min \
    --model-file ./experiments/inspired_blender_baseline/model/blender_0202 \
    -eps 3.0