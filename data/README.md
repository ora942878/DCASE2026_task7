# Data Folder

Place all local datasets under this folder. The audio data is not included in
the GitHub repository, but the scripts expect the following relative paths.

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

## What Each Folder Is For

- `D2/`: development Domain 2 data. The D2/D3 validation script reads
  `D2/d2-dev-test/*.wav` and `D2/metadata/d2-dev-test.csv`.
- `D3/`: development Domain 3 data. The D2/D3 validation script reads
  `D3/d3-dev-test/*.wav` and `D3/metadata/d3-dev-test.csv`.
- `eval/`: official evaluation wav files. The final official inference script
  reads all wav files directly from `eval/*.wav` and writes the prediction file
  under `code/final_eval/submission/Gao_SHNU_task7_1/`.

## Metadata

The development metadata CSV files should contain at least these columns:

```text
filename,class
```

`filename` should match the wav filename in the corresponding split folder, and
`class` should be one of the Task 7 class labels.

The official `eval/` folder does not need labels or metadata for inference.

## Audio

- Format: `.wav`
- Sample rate: 32 kHz
- Channels: mono is preferred; stereo is averaged to mono by the loaders
