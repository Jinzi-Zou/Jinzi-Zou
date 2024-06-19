cd ../../../src/
python -u enhance_dataprocess.py \
    --dataset citeseer \
    --model gcn \
    --p-model gcn \
    --p-loss AUC+CE \
    --eval-steps 1 \
    --log-steps 100 \
    --epochs 500 \
    --runs 5 \
    --negative-sampler global \
    --eval-metric hits \
    --lr 0.005 \
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
    --q 0.1 \
    --n-neg 1 \
    --auxiliary_size 900 \
    --addtrain \
    --batch-size 65536 \
    --bn

    #750