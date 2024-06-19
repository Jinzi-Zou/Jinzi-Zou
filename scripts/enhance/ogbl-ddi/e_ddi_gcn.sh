cd ../../../src/
python -u enhance_dataprocess.py \
    --dataset ogbl-ddi \
    --model gcn \
    --p-model gcn\
    --p-loss CE\
    --eval-steps 1 \
    --log-steps 100 \
    --epochs 600 \
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
    --n-neg 1 \
    --q 0.4 \
    --auxiliary_size 530000\
    --addtrain \
    --bn


