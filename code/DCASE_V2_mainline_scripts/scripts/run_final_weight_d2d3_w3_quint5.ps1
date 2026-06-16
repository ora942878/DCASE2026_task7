$ErrorActionPreference = "Stop"

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
$RepoRoot = Split-Path -Parent (Split-Path -Parent $Root)

$Out = "final_weight_d2d3_w3_quint5"
$Checkpoint = Join-Path $RepoRoot "checkpoints\ours\Gao_SHNU_task7_1_D3_dictionary.pth"
$D2Test = Join-Path $RepoRoot "data\D2\d2-dev-test"
$D3Test = Join-Path $RepoRoot "data\D3\d3-dev-test"
$RunDir = Join-Path $Root "runs\$Out"

if (-not (Test-Path $Checkpoint)) {
    throw "Missing checkpoint: $Checkpoint"
}
if (-not (Test-Path $D2Test)) {
    throw "Missing D2 test wav directory: $D2Test"
}
if (-not (Test-Path $D3Test)) {
    throw "Missing D3 test wav directory: $D3Test"
}

New-Item -ItemType Directory -Path $RunDir -Force | Out-Null

Push-Location $Root
try {
    & $Python -m py_compile "scripts\eval_final_inference_ablation.py"
    & $Python "scripts\eval_final_inference_ablation.py" `
        --checkpoint $Checkpoint `
        --out-name $Out `
        --task-id 2 `
        --no-full-clip `
        --window-sec 3.0 `
        --center-sets "quint5:0.1,0.3,0.5,0.7,0.9" `
        --batch-size 256 |
        Tee-Object -FilePath (Join-Path $RunDir "pipeline.log")

    Write-Host ""
    Write-Host "Summary:"
    Get-Content (Join-Path $RunDir "summary.csv")
}
finally {
    Pop-Location
}
