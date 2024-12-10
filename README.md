# SSPO
Self-supervised Transformer-based Online Portfolio Optimizer
![图片1](https://github.com/user-attachments/assets/5b8e675d-e995-47a1-9222-573927d6f11a)



# Step 1
Run datahandler.py

# Step 2
Run train.py

# (Optional) DDP Training

torchrun --nproc_per_node=2 train_ddp.py

# Tensorboard
tensorboard --logdir=runs/experiment0
