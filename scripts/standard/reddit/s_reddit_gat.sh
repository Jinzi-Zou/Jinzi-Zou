cd ../../../src/
python enhance_generate.py \
    --dataset reddit \
    --model gat \
    --eval-steps 1 \
    --log-steps 100 \
    --epochs 500 \
    --runs 10\
    --negative-sampler global \
    --eval-metric hits \
    --lr 0.0001 \
    --hop-norm \
    --n-layers 2 \
    --n-hidden 512 \
    --out-hidden 512 \
    --n-heads 1 \
    --dropout 0.5 \
    --attn-drop 0. \
    --input-drop 0. \
    --diffusion-drop 0. \
    --loss-func AUC \
    --n-neg 1 \
    --bn