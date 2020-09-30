#!/bin/bash
# train inspired response generator (gpt2 based model)
set -xue


# parlai train_model  -m hugging_face/gpt2 \
#     -t inspired_response_generator \
#     --add-special-tokens True \
#     --add-start-token True \
#     --gpt2-size small \
#     -eps 20.0 \
#     -bs 1 \
#     -opt adam \
#     -lr 1e-3 \
#     --add_inspired_special_tokens True  \
#     -mf ./experiments/inspired_response_generator/model/0907_exp1_initial

# parlai train_model  -m hugging_face/gpt2 \
#     -t inspired_response_generator \
#     --add-special-tokens True \
#     --add-start-token True \
#     --gpt2-size small \
#     -eps 10.0 \
#     -bs 1 \
#     -opt adam \
#     -lr 1e-3 \
#     --add_inspired_special_tokens True  \
#     --validation-patience 5 \
#     --validation-metric ppl \
#     --validation-metric-mode min \
#     -mf ./experiments/inspired_response_generator/model/0908_exp2_conditional_tokens

# parlai train_model  -m hugging_face/gpt2 \
#     -t inspired_response_generator \
#     --add-special-tokens True \
#     --add-start-token True \
#     --gpt2-size small \
#     -eps 10.0 \
#     -bs 1 \
#     -opt adam \
#     -lr 1e-3 \
#     --add_inspired_special_tokens True  \
#     --validation-patience 2 \
#     --validation-metric ppl \
#     --validation-metric-mode min \
#     -mf ./experiments/inspired_response_generator/model/redial_pretrain_conditional_placeholders



# parlai train_model  -m hugging_face/gpt2 \
#     --init-model ./experiments/inspired_response_generator/model/redial_pretrain_conditional_placeholders \
#     -t inspired_response_generator \
#     --add-special-tokens True \
#     --add-start-token True \
#     --gpt2-size small \
#     -eps 10.0 \
#     -bs 1 \
#     -opt adam \
#     -lr 1e-3 \
#     --add_inspired_special_tokens True  \
#     --validation-patience 2 \
#     --validation-metric ppl \
#     --validation-metric-mode min \
#     -mf ./experiments/inspired_response_generator/model/redial_pretrain_conditional_placeholders_exp3_epoch10

parlai multiprocessing_train  -m hugging_face/gpt2 \
    -t inspired_response_generator \
    --add-special-tokens True \
    --add-start-token True \
    --gpt2-size small \
    -eps 3.0 \
    -bs 1 \
    -opt adam \
    -lr 1e-3 \
    --add_inspired_special_tokens True  \
    --validation-patience 2 \
    --validation-metric ppl \
    --validation-metric-mode min \
    -mf ./experiments/inspired_response_generator/model/redial_pretrain_model2_conditional_placeholders