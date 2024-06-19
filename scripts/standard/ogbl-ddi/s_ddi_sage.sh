cd ../../../src/
python -u enhance_generate.py \
    --dataset ogbl-ddi \
    --model sage \
    --eval-steps 1 \
    --log-steps 100 \
    --epochs 1000 \
    --runs 10\
    --negative-sampler global \
    --eval-metric hits \
    --lr 0.001 \
    --hop-norm \
    --n-layers 2 \
    --n-hidden 512 \
    --out-hidden 512 \
    --n-heads 1 \
    --dropout 0.5 \
    --attn-drop 0. \
    --input-drop 0. \
    --diffusion-drop 0. \
    --loss-func CE \
    --n-neg 1 \
    --bn


