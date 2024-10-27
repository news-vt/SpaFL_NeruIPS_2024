# SpaFL - NeruIPS 2024
# 🚧 **UNDER CONSTRUCTION** 🚧
This is a repository for the paper "SpaFL: Communication-Efficient Federated Learning with Sparse Models and Low Computational Overhead?" published in NeurIPS 2024

# Install #
1. create a conda env with python=3.10.14
2. ```pip install -r requirements.txt```
3. ```pip install --upgrade protobuf wandb```

# Run #
1. FMNIST: ```python train.py --learning_rate 0.001 --th_coeff 0.002 --local_epoch 5 --seed 1```
2. CIFAR-10: ```python train.py --affix grad_clip_3_seed_1_conv4 --model conv4 --learning_rate 0.01 --th_coeff 0.00015 --batch_size 16 --alpha 0.1 --local_ep 5 --clip 3 --seed 1```
3. CIFAR-10 with ViT: ```python train.py --model vit --batch_size 16 --comm_rounds 500 --learning_rate 0.01 --seed 1 --alpha 0.1 --th_coeff 0.0001 --local_epoch 1 --clip 3```
4. CIFAR-100: ```python train.py --model resnet18 --comm_rounds 1500 --learning_rate 0.01 --lr_decay 0.993 --seed 1 --th_coeff 0.0007 --local_epoch 7 --clip 15 --seed 1```
