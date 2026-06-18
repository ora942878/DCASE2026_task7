# Ablation 03: Final Inference Window

This folder is for the final inference-window ablation.

It compares full-clip inference with multi-center window inference, including the 3s, 4s, and 5s window settings used in the technical report.

This ablation should be run after the main D1 -> D2 -> D3 training pipeline has produced the final sequential model. Experimentally, it explains why the final system uses a fixed short-window inference strategy instead of directly evaluating the whole clip.
