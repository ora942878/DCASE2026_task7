# Ablation 01: Data Augmentation

This folder isolates the effect of the data augmentation policy.

All variants use the same D2+D3 mixed fine-tuning setting and start from the same official checkpoint1 with BN1. The changed factor is only the training view: plain data, self concat, same-class concat, shift/gain, and the final same-class concat + shift/gain view.

Experimentally, this ablation answers whether the proposed stochastic augmentation improves the D2/D3 average wav-level macro recall over plain fine-tuning.
