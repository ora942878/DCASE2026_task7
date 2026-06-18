# Ablation 02: Multi-path Beta Update

This folder studies how stochastic augmentation is converted into a checkpoint update.

Five D3 augmentation views are generated with different random seeds, then five fine-tuning paths are trained from the official checkpoint2 with BN2. Their parameter updates are averaged and applied with a beta scale:

```text
theta_beta = theta_old + beta * mean(theta_path_i - theta_old)
```

Experimentally, beta = 1 represents the pure five-path mean update, while beta < 1 keeps part of the previous checkpoint to balance D2 retention and D3 adaptation.
