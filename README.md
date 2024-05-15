# ReDiffuser
**ReDiffuser: Reliable Decision-Making Using a Diffuser with Confidence Estimation**

The original implementation of ReDiffuser accepted by ICML 2024.

![framework](https://github.com/he-nantian/ReDiffuser/blob/main/framework.png)

## 1. Download checkpoints

We have uploaded the checkpoints of Diffuser and RND along with the generated dataset used to train RND to the [google drive](https://drive.google.com/drive/folders/1AjXwI5XOYiyEoXqWiGozLztPjaqIlz7T?usp=sharing). Note that the RND dataset in Maze2D environment is separately stored in the [baidu drive](https://pan.baidu.com/s/1YOn9UH188UKzXrtQy9KKgw?pwd=xn64) with the extraction code as xn64 due to the storage limitation of google drive.

## 2. Create virtual environment

<code>conda env create -f environment.yml</code>

## 3. Run scripts

(1) To make experiments in one of the three tasks, run:

<code>cd \<task name\></code>

(2) To pretrain the Diffuser model, run:

<code>python scripts/train.py</code>

(3) To generate the dataset used to train the RND model, run:

<code>python scripts/generate_dataset.py</code>

(4) To train the RND model, run:

<code>python scripts/rnd_train.py</code>

(5) To make reliable planning, run:

<code>python scripts/rnd_plan.py</code>

## *Hyperparameters*

Most of the hyperparameters are specified in the *config* folder, and some of the hyperparameters are specified in the code by using argparse. It is important to know the meaning of each hyperparameter, and feel free to raise an issue if you get puzzled.

## Acknowledgements

The Diffuser part is implemented based on [Janner's Diffuser repository](https://github.com/jannerm/diffuser); The RND part is implemented based on [Pytorch implementation of RND](https://github.com/jcwleo/random-network-distillation-pytorch).