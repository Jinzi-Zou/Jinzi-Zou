cd ../../../src/
python -u enhance_dataprocess.py \
    --dataset pubmed \
    --model agdn \
    --p-model agdn \
    --p-loss CE \
    --eval-steps 1 \
    --log-steps 100 \
    --epochs 1000 \
    --runs 5 \
    --negative-sampler global \
    --eval-metric hits \
    --lr 0.001 \
    --K 2 \
    --hop-norm \
    --n-layers 2 \
    --n-hidden 512 \
    --out-hidden 512 \
    --n-heads 1 \
    --dropout 0.3 \
    --attn-drop 0. \
    --input-drop 0. \
    --diffusion-drop 0. \
    --loss-func RoL \
    --q 0.1 \
    --n-neg 1 \
    --auxiliary_size 8600 \
    --addtrain \
    --batch-size 65536 \
    --bn


    #7000