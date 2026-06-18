# Final Official Eval Inference

This folder contains the final Task 7 evaluation-set inference code.

## Repository-Relative Defaults

- Eval wav directory: `../../data/eval`
- Final D3 checkpoint: `../../checkpoints/ours/Gao_SHNU_task7_1_D3_dictionary.pth`
- Official-format output: `submission/Gao_SHNU_task7_1/Gao_SHNU_task7_1.output.csv`
- Manifest: `submission/Gao_SHNU_task7_1/Gao_SHNU_task7_1_manifest.json`
- Preview CSV: `submission/Gao_SHNU_task7_1/Gao_SHNU_task7_1_preview.csv`

The script resolves these paths from its own location, so the repository can be moved without editing path constants.

## Fixed Method

- Submission label: `Gao_SHNU_task7_1`
- BN branch: `task_id = 2`
- Window length: `3.0` seconds
- Centers: `0.1, 0.3, 0.5, 0.7, 0.9`
- Aggregation: choose the prediction from the window with the highest softmax confidence
- Output format: no header, one row per wav:

```text
filename.wav    class_label
```

This fixed `task_id = 2` setting is for the official unlabeled eval-set submission with the final D3 dictionary. It is separate from D2/D3 development-set analyses that compare different BN/task-id evaluation protocols.

## Run

From the repository root:

```bash
python code/final_eval/predict_official_eval.py
```

On Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\code\final_eval\run_final_eval.ps1
```

The launcher uses `python` by default. Set `PYTHON` if you want a specific interpreter.

## Local Runnable Frontend

`domain_net.py` keeps the same CNN14-style model body and domain-specific BatchNorm branches as the training artifact, but replaces the fixed `torchlibrosa`/`librosa` frontend with a local `torch.stft` + built-in Slaney mel-filter implementation. This avoids frontend initialization hangs observed in one Windows conda environment.

When loading `Gao_SHNU_task7_1_D3_dictionary.pth`, the script still loads the trained model weights and only ignores the obsolete `spectrogram_extractor.stft.*` frontend convolution matrices from the original torchlibrosa implementation. Any other checkpoint mismatch raises an error.
