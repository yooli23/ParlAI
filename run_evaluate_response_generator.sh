#!/bin/bash
# evaluate inspired response generator (gpt2 based model)
set -xue

parlai eval_model -t inspired_response_generator \
    -mf "./experiments/inspired_response_generator/model/blender_1110_exp7" \
    -dt test \
    --skip-generation False \
    --inference nucleus \
    --metrics bleu \
    
    # --add-special-tokens True