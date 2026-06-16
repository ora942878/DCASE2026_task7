$ErrorActionPreference = "Stop"

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeRoot = Split-Path -Parent $Here
$ReleaseRoot = Split-Path -Parent $CodeRoot
$SubmissionLabel = "Gao_SHNU_task7_1"
$SubmissionDir = Join-Path $Here "submission\$SubmissionLabel"

& $Python (Join-Path $Here "predict_official_eval.py") `
  --audio-dir (Join-Path $ReleaseRoot "data\eval") `
  --checkpoint (Join-Path $ReleaseRoot "checkpoints\ours\$SubmissionLabel`_D3_dictionary.pth") `
  --output-csv (Join-Path $SubmissionDir "$SubmissionLabel.output.csv") `
  --manifest (Join-Path $SubmissionDir "$SubmissionLabel`_manifest.json") `
  --preview-csv (Join-Path $SubmissionDir "$SubmissionLabel`_preview.csv") `
  --task-id 2 `
  --window-sec 3.0 `
  --centers "0.1,0.3,0.5,0.7,0.9" `
  --batch-size 256
