set -xu

# parlai display_model -t redial \
#     --model-file ./experiments/inspired_blender_redial/model/blender_redial_0202 \
#     --skip-generation False \
#     -dt test \
#     -n 500 > redial_res.txt

parlai display_model -t inspired_blender_baseline \
    --model-file ./experiments/inspired_blender_baseline/model/blender_0202 \
    --skip-generation False \
    -dt test \
    -n 500 > inspired_res.txt
