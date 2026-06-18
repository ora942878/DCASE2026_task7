# DCASE Task 7

Code for DCASE Task 7 final inference, mainline training, and ablation experiments.

Datasets and `.pth` checkpoints are not included in GitHub. Put them under
`data/` and `checkpoints/` before running the scripts.

## Structure

```text
.
|-- code/
|   |-- final_eval/                 # official eval-set inference
|   |-- base/                       # shared model, configs, and dataset utilities
|   `-- DCASE_V2_mainline_scripts/
|       |-- main_pipeline/          # modular D1 -> D2 -> D3 training pipeline
|       |-- ablation01_prepare_views/
|       |-- ablation01_data_augmentation/
|       |-- ablation02_multipath_beta/
|       `-- ablation03_inference_window/
|-- data/                           # local datasets, not tracked by Git
|-- checkpoints/                    # local checkpoints, not tracked by Git
|-- requirements.txt
`-- README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

Expected layout:

```text
data/
|-- D2/
|   |-- d2-dev-train/*.wav
|   |-- d2-dev-test/*.wav
|   `-- metadata/
|       |-- d2-dev-train.csv
|       `-- d2-dev-test.csv
|-- D3/
|   |-- d3-dev-train/*.wav
|   |-- d3-dev-test/*.wav
|   `-- metadata/
|       |-- d3-dev-train.csv
|       `-- d3-dev-test.csv
`-- eval/
    `-- *.wav
```

See `data/README.md` for the same layout in the data folder.

## Checkpoints

Checkpoint download links:

```text
Gao_SHNU_task7_1_D2_dictionary.pth: https://drive.google.com/file/d/1y00TLlw0vAc0XOPMyM0uSz5zYzEIjNvP/view?usp=sharing
Gao_SHNU_task7_1_D3_dictionary.pth: https://drive.google.com/file/d/1hdaG1_HZBktGDEuiJKCwroHUP_0AqIVC/view?usp=sharing
```

For GitHub-style local reproduction:

```text
checkpoints/ours/
|-- Gao_SHNU_task7_1_D2_dictionary.pth
`-- Gao_SHNU_task7_1_D3_dictionary.pth
```

The final eval script reads the D3 checkpoint from `checkpoints/ours/` by
default and writes official-format predictions under `code/final_eval/submission/`.

## Final Eval Inference

From the repository root:

```bash
PYTHON=/path/to/python bash code/final_eval/run_final_eval.sh
```

Windows PowerShell:

```powershell
$env:PYTHON="path\to\python.exe"
powershell -ExecutionPolicy Bypass -File .\code\final_eval\run_final_eval.ps1
```

Output:

```text
code/final_eval/submission/Gao_SHNU_task7_1/Gao_SHNU_task7_1.output.csv
```

## D2/D3 Validation

The development-set validation and inference-window ablation are under
`code/DCASE_V2_mainline_scripts/ablation03_inference_window/`.

Windows PowerShell:

```powershell
$env:PYTHON="path\to\python.exe"
powershell -ExecutionPolicy Bypass -File .\code\DCASE_V2_mainline_scripts\ablation03_inference_window\run_final_weight_d2d3_w3_quint5.ps1
```

Bash:

```bash
PYTHON=/path/to/python bash code/DCASE_V2_mainline_scripts/ablation03_inference_window/run_final_weight_d2d3_w3_quint5.sh
```

This evaluates the final D3 checkpoint on `data/D2/d2-dev-test/` and
`data/D3/d3-dev-test/`.

## Mainline And Ablations

The main training and ablation code is organized as small modules:

```text
code/DCASE_V2_mainline_scripts/
|-- main_pipeline/
|   |-- train/run_final_pipeline.py
|   |-- tools/
|   `-- inference/eval_c2c3_on_d2d3.py
|-- ablation01_prepare_views/
|-- ablation01_data_augmentation/
|-- ablation02_multipath_beta/
`-- ablation03_inference_window/
```

Useful entry points:

```bash
python code/DCASE_V2_mainline_scripts/main_pipeline/train/run_final_pipeline.py --help
python code/DCASE_V2_mainline_scripts/ablation01_prepare_views/build_ablation_views.py --help
python code/DCASE_V2_mainline_scripts/ablation01_data_augmentation/train_ablation_mixed_ft.py --help
python code/DCASE_V2_mainline_scripts/ablation02_multipath_beta/sweep_beta.py --help
python code/DCASE_V2_mainline_scripts/ablation03_inference_window/eval_final_inference_ablation.py --help
```

Each ablation folder contains a short README describing its experimental role.

## Notes

- Official output rows use four spaces between filename and label:

```text
filename.wav    class_label
```

