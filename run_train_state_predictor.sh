#!/bin/bash
# train inspired state predictor (gpt2 based model)
set -xue

# ut + bt -> diff, all training data
# parlai train_model  -m hugging_face/gpt2 \
#     -t inspired_state_predictor \
#     --add-special-tokens True \
#     --add-start-token True \
#     --gpt2-size medium \
#     -eps 20.0 \
#     -bs 1 \
#     -opt adam \
#     -lr 1e-3 \
#     --add_inspired_special_tokens True  \
#     --validation-patience 5 \
#     -mf ./experiments/inspired_state_predictor/model/0907_exp3_diff_label

# context + [SEP] + ut + [SEP] + bt -> diff, cut off data when diff is None for 3 consecutive turns
# parlai train_model  -m hugging_face/gpt2 \
#     -t inspired_state_predictor \
#     --add-special-tokens True \
#     --add-start-token True \
#     --gpt2-size small \
#     -eps 20.0 \
#     -bs 1 \
#     -opt adam \
#     -lr 1e-3 \
#     --add_inspired_special_tokens True  \
#     --validation-patience 5 \
#     -mf ./experiments/inspired_state_predictor/model/0907_exp4_context_diff_label

# context + [SEP] + ut + [SEP] + bt -> diff, cut off data when diff is None for 3 consecutive turns
# change label to tokens instead of list ['[movie_genre_0]'] -> [movie_genre_0]
# parlai train_model  -m hugging_face/gpt2 \
#     -t inspired_state_predictor \
#     --add-special-tokens True \
#     --add-start-token True \
#     --gpt2-size small \
#     -eps 20.0 \
#     -bs 1 \
#     -opt adam \
#     -lr 1e-3 \
#     --add_inspired_special_tokens True  \
#     --validation-patience 5 \
#     -mf ./experiments/inspired_state_predictor/model/0907_exp5_context_diff_label

# # bt -> diff
# parlai train_model  -m hugging_face/gpt2 \
#     -t inspired_state_predictor \
#     --add-special-tokens True \
#     --add-start-token True \
#     --gpt2-size small \
#     -eps 20.0 \
#     -bs 1 \
#     -opt adam \
#     -lr 1e-3 \
#     --add_inspired_special_tokens True  \
#     --validation-patience 5 \
#     -mf ./experiments/inspired_state_predictor/model/0907_exp6_bt_diff_label

# bt -> bt+1
# parlai train_model  -m hugging_face/gpt2 \
#     -t inspired_state_predictor \
#     --add-special-tokens True \
#     --add-start-token True \
#     --gpt2-size small \
#     -eps 20.0 \
#     -bs 1 \
#     -opt adam \
#     -lr 1e-3 \
#     --add_inspired_special_tokens True  \
#     --validation-patience 5 \
#     -mf ./experiments/inspired_state_predictor/model/0907_exp7_bt_btplus1_label

# context + bt -> bt+1
# parlai train_model  -m hugging_face/gpt2 \
#     -t inspired_state_predictor \
#     --add-special-tokens True \
#     --add-start-token True \
#     --gpt2-size small \
#     -eps 20.0 \
#     -bs 1 \
#     -opt adam \
#     -lr 1e-3 \
#     --add_inspired_special_tokens True  \
#     --validation-patience 5 \
#     -mf ./experiments/inspired_state_predictor/model/0907_exp8_context_and_bt_btplus1_label

# 0920 new data bt -> bt+1
# parlai train_model  -m hugging_face/gpt2 \
#     -t inspired_state_predictor \
#     --add-special-tokens True \
#     --add-start-token True \
#     --gpt2-size small \
#     -eps 20.0 \
#     -bs 1 \
#     -opt adam \
#     -lr 1e-3 \
#     --add_inspired_special_tokens True  \
#     --validation-patience 3 \
#     -mf ./experiments/inspired_state_predictor/model/0921_exp9_bt_btplus1_label


# # 0920 bt -> diff
# parlai train_model  -m hugging_face/gpt2 \
#     -t inspired_state_predictor \
#     --add-special-tokens True \
#     --add-start-token True \
#     --gpt2-size small \
#     -eps 20.0 \
#     -bs 1 \
#     -opt adam \
#     -lr 1e-3 \
#     --add_inspired_special_tokens True  \
#     --validation-patience 3 \
#     -mf ./experiments/inspired_state_predictor/model/0907_exp10_bt_diff_label

# 0920 bt -> related placeholder
# parlai train_model  -m hugging_face/gpt2 \
#     -t inspired_state_predictor \
#     --add-special-tokens True \
#     --add-start-token True \
#     --gpt2-size small \
#     -eps 10.0 \
#     -bs 1 \
#     -opt adam \
#     -lr 1e-3 \
#     --add_inspired_special_tokens True  \
#     --validation-patience 3 \
#     -mf ./experiments/inspired_state_predictor/model/0921_exp11_bt_relatedplacehoders_label

# 0920 data bt -> related placeholder, remove placeholders -> chitchat, leave chitchat -> chitchat
# parlai train_model  -m hugging_face/gpt2 \
#     -t inspired_state_predictor \
#     --add-special-tokens True \
#     --add-start-token True \
#     --gpt2-size small \
#     -eps 10.0 \
#     -bs 1 \
#     -opt adam \
#     -lr 1e-3 \
#     --add_inspired_special_tokens True  \
#     --validation-patience 3 \
#     -mf ./experiments/inspired_state_predictor/model/0921_exp12_bt_relatedplacehoders_label

# 0929 user_utterance + bt -> related placeholder leave chitchat -> chitchat.
parlai train_model  -m hugging_face/gpt2 \
    -t inspired_state_predictor \
    --add-special-tokens True \
    --add-start-token True \
    --gpt2-size small \
    -eps 10.0 \
    -bs 1 \
    -opt adam \
    -lr 1e-3 \
    --add_inspired_special_tokens True  \
    --validation-patience 3 \
    -mf ./experiments/inspired_state_predictor/model/0929_exp13_ut_bt_relatedplacehoders_label