# SSPO
Self-supervised Transformer-based Online Portfolio Optimizer
![image](https://github.com/user-attachments/assets/935cd6fa-8e01-4048-82e3-9134bf5b796b)




# Step 1
Run datahandler.py

# Step 2
Run train.py

# (Optional) DDP Training

torchrun --nproc_per_node=2 train_ddp.py

# Tensorboard
tensorboard --logdir=runs/experiment0
