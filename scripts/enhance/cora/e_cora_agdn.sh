cd ../../../src/
python -u enhance_dataprocess.py \
    --dataset cora \
    --model agdn \
    --p-model agdn \
    --p-loss CE \
    --eval-steps 1 \
    --log-steps 100 \
    --epochs 200 \
    --runs 10 \
    --negative-sampler global \
    --eval-metric hits \
    --lr 0.001 \
    --K 2 \
    --hop-norm \
    --n-layers 2 \
    --n-hidden 256 \
    --out-hidden 128 \
    --n-heads 1 \
    --dropout 0.1 \
    --attn-drop 0. \
    --input-drop 0. \
    --diffusion-drop 0. \
    --loss-func RoL \
    --q 0.2 \
    --n-neg 1 \
    --auxiliary_size 1300\
    --batch-size 1024 \
    --addtrain \
    --bn

    #820