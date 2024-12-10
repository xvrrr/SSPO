# SSPO
Self-supervised Transformer-based Online Portfolio Optimizer
![image](https://github.com/user-attachments/assets/37bb5108-bbed-45e4-b52a-f0d41a1caf6e)


# Step 1
Run datahandler.py

# Step 2
Run train.py

# (Optional) DDP Training

torchrun --nproc_per_node=2 train_ddp.py

# Tensorboard
tensorboard --logdir=runs/experiment0
