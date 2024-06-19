cd ../../../src/
python -u enhance_dataprocess.py \
    --dataset email \
    --model gat \
    --p-model gat\
    --p-loss AUC+CE\
    --eval-steps 1 \
    --log-steps 300 \
    --epochs 600 \
    --runs 10\
    --negative-sampler global \
    --eval-metric hits \
    --lr 0.0001 \
    --hop-norm \
    --n-layers 2 \
    --n-hidden 512 \
    --out-hidden 512 \
    --n-heads 1 \
    --dropout 0. \
    --attn-drop 0. \
    --input-drop 0. \
    --diffusion-drop 0. \
    --loss-func RoL \
    --q 0.3 \
    --n-neg 1 \
    --auxiliary_size 3000 \
    --batch-size 12288 \
    --addtrain \
    --bn