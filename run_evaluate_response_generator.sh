#!/bin/bash
# evaluate inspired response generator (gpt2 based model)
set -xue

parlai display_model -t inspired_response_generator \
    -mf "./experiments/inspired_response_generator/model/blender_0929_exp1" \
    -dt test \
    -n 100 \
    --skip-generation False \
    --inference nucleus \
    
    # --add-special-tokens True