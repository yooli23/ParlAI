#!/bin/bash
# train inspired state predictor (gpt2 based model)
set -xue

parlai display_model -t inspired_state_predictor \
    -mf "./experiments/inspired_state_predictor/model/0929_exp13_ut_bt_relatedplacehoders_label" \
    -dt test \
    -n 100 \
    --add-special-tokens True \
    --inference nucleus