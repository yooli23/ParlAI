#!/bin/bash
# evaluate inspired response generator (gpt2 based model)
set -xue

parlai display_model -t inspired_response_generator \
    -mf "./experiments/inspired_response_generator/model/0908_exp2_conditional_tokens" \
    -dt test \
    -n 100 \
    --add-special-tokens True