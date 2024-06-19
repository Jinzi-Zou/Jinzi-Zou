cd ../../../src/
python -u enhance_dataprocess.py \
    --dataset reddit \
    --model agdn \
    --p-model agdn\
    --p-loss CE\
    --K 3 \
    --eval-steps 1 \
    --log-steps 100 \
    --epochs 800 \
    --runs 5\
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
    --loss-func RoL \
    --q 0.1 \
    --n-neg 1 \
    --auxiliary_size 52000\
    --addtrain \
    --bn