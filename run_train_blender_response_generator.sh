set -xue

# blender based response generator
# parlai multiprocessing_train -t inspired_response_generator \
#     -m transformer/generator \
#     --multitask-weights 1,3,3,3 \
#     --init-model zoo:blender/blender_3B/model \
#     --dict-file zoo:blender/blender_3B/model.dict \
#     --embedding-size 512 \
#     --n-layers 8 \
#     --ffn-size 2048 \
#     --dropout 0.1 \
#     --n-heads 16 \
#     --learn-positional-embeddings True \
#     --n-positions 512 \
#     --variant xlm \
#     --activation gelu \
#     --skip-generation True \
#     --fp16 True \
#     --text-truncate 512 \
#     --label-truncate 128 \
#     --dict-tokenizer bpe \
#     --dict-lower True \
#     -lr 1e-06 \
#     --optimizer adamax \
#     --lr-scheduler reduceonplateau \
#     --gradient-clip 0.1 \
#     -veps 0.25 \
#     --betas 0.9,0.999 \
#     --update-freq 1 \
#     --attention-dropout 0.0 \
#     --relu-dropout 0.0 \
#     --skip-generation True \
#     -vp 15 \
#     -stim 60 \
#     -vme 20000 \
#     -bs 4 \
#     -vmt ppl \
#     -vmm min \
#     --save-after-valid True \
#     --model-file ./experiments/inspired_response_generator/model/blender_0929_exp1\
#     -eps 10.0


parlai multiprocessing_train -t inspired_response_generator \
    --multitask-weights 1,3,3,3 \
    -veps 0.25 \
    --attention-dropout 0.0 \
    --batchsize 1 \
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
    --fp16 False \
    --init-model zoo:blender/blender_3B/model \
    --dict-file zoo:blender/blender_3B/model.dict \
    --label-truncate 128 \
    --log_every_n_secs 10 \
    -lr 7e-06 \
    --lr-scheduler reduceonplateau \
    --lr-scheduler-patience 3 \
    --optimizer adam \
    --relu-dropout 0.0 \
    --activation gelu \
    --model-parallel false \
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
    --model-file ./experiments/inspired_response_generator/model/blender_0929_exp2_3B_model \
    -eps 10.0