# DCASE Task 7 Mainline

Code for DCASE Task 7 inference and development-set validation.

Datasets and `.pth` checkpoints are not included in GitHub. Put them under
`data/` and `checkpoints/` before running the scripts.

## Structure

```text
.
|-- code/
|   |-- final_eval/                 # final official eval inference
|   |-- DCASE_V2_mainline_scripts/  # D2/D3 validation and experiment scripts
|   |-- DCASE_CODE_V2/              # base model/training code
|   `-- submission_clean/           # generic inference wrapper
|-- data/                           # local datasets, not tracked by Git
|-- checkpoints/                    # local checkpoints, not tracked by Git
|-- requirements.txt
`-- README.md
```

## Setup

```bash
pip install -r requirements.txt
```

If using GPU, install the PyTorch build that matches your CUDA environment.

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
Gao_SHNU_task7_1_D2_dictionary.pth: TODO
Gao_SHNU_task7_1_D3_dictionary.pth: TODO
```

For GitHub-style local reproduction:

```text
checkpoints/ours/
|-- Gao_SHNU_task7_1_D2_dictionary.pth
`-- Gao_SHNU_task7_1_D3_dictionary.pth
```

For the official DCASE submission package, also place the same two checkpoint
files next to `Gao_SHNU_task7_1_model.py` in:

```text
code/final_eval/submission/Gao_SHNU_task7_1/
```

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

Windows PowerShell:

```powershell
$env:PYTHON="path\to\python.exe"
powershell -ExecutionPolicy Bypass -File .\code\DCASE_V2_mainline_scripts\scripts\run_final_weight_d2d3_w3_quint5.ps1
```

Bash:

```bash
PYTHON=/path/to/python bash code/DCASE_V2_mainline_scripts/scripts/run_final_weight_d2d3_w3_quint5.sh
```

This evaluates the final D3 checkpoint on `data/D2/d2-dev-test/` and
`data/D3/d3-dev-test/`.

## Notes

- Official output rows use four spaces between filename and label:

```text
filename.wav    class_label
```

- Historical ablation launchers are kept in
  `code/DCASE_V2_mainline_scripts/scripts/legacy_or_ablation/`.
