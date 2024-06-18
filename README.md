# RoGALP: Robust graph augmentation for link prediction

## Requirements

- dgl==0.8.1
- gensim==4.0.1
- networkx==2.6.3
- numpy==1.21.2
- pytorch==1.10.1
- torch-geometric==2.0.4

To install all dependencies:

```setup
pip install -r requirements.txt
```

## Training

To train the model in the paper, run this command （here we provide three examples）：

```train
python -u enhance_dataprocess.py --dataset email --model gcn --p-model gcn --p-loss AUC+CE --eval-steps 1 --log-steps 600 --epochs 600 --runs 1 --negative-sampler global --eval-metric hits --lr 0.0001 --hop-norm --n-layers 2 --n-hidden 512 --out-hidden 512 --n-heads 1 --dropout 0. --attn-drop 0. --input-drop 0. --diffusion-drop 0. --loss-func RoL --q 0.4 --n-neg 1 --auxiliary_size 4000 --addtrain --batch-size 12288 --bn
```

```
python -u enhance_dataprocess.py --dataset cora --model gcn --p-model gcn --p-loss CE --eval-steps 1 --log-steps 200 --epochs 200 --runs 10 --negative-sampler global --eval-metric hits --lr 0.01 --hop-norm --n-layers 2 --n-hidden 256 --out-hidden 128 --n-heads 1 --dropout 0.5 --attn-drop 0. --input-drop 0. --diffusion-drop 0. --loss-func RoL --q 0.3 --n-neg 1 --auxiliary_size 1300 --addtrain --batch-size 1024 --bn
```

The training commands for other datasets are summarized in the "scripts" folder.

