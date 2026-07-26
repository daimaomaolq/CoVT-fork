param(
    [Parameter(Mandatory = $true)][string]$Index,
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [string]$AdapterPath = "",
    [string]$InitialPredictions = "",
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [string]$Python = "python",
    [string]$Device = "auto",
    [string]$TorchDtype = "auto",
    [string]$FeedbackMode = "template",
    [string]$AnchorModelId = "['sam','dino']",
    [string]$AnchorPromptMode = "query_tail",
    [string]$AnchorTokenCounts = "",
    [int]$Limit = 0,
    [bool]$Resume = $true
)

$ErrorActionPreference = "Stop"
if (-not $env:PYTORCH_CUDA_ALLOC_CONF) {
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
}
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$evalScript = Join-Path $repoRoot "train/src/tools/eval_dvgbench_agentic_v3.py"
$summaryScript = Join-Path $repoRoot "train/src/tools/summarize_dvgbench_agentic_v3_matrix.py"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

function Invoke-AgenticExperiment {
    param(
        [string]$Name,
        [string]$Method,
        [int]$Budget,
        [string[]]$Disabled = @(),
        [string]$RunFeedbackMode = "off",
        [string[]]$ExtraArgs = @()
    )
    $prediction = Join-Path $OutputDir "$Name.jsonl"
    $summary = Join-Path $OutputDir "$Name.summary.json"
    if ($Resume -and (Test-Path $prediction) -and (Test-Path $summary)) {
        Write-Host "[agentic-matrix] skip completed $Name"
        return
    }
    $arguments = @(
        $evalScript,
        "--index", $Index,
        "--output", $prediction,
        "--summary-output", $summary,
        "--query-field", "question",
        "--method", $Method,
        "--model-path", $ModelPath,
        "--device", $Device,
        "--torch-dtype", $TorchDtype,
        "--anchor-model-id", $AnchorModelId,
        "--anchor-prompt-mode", $AnchorPromptMode,
        "--max-specialized-unit-calls", "$Budget",
        "--feedback-mode", $RunFeedbackMode
    )
    $arguments += $ExtraArgs
    if ($Resume) {
        $arguments += "--resume"
    }
    if ($AdapterPath) {
        $arguments += @("--adapter-path", $AdapterPath)
    }
    if ($InitialPredictions) {
        $arguments += @("--initial-predictions", $InitialPredictions)
        $arguments += "--require-initial-confidence"
    }
    if ($Limit -gt 0) {
        $arguments += @("--limit", "$Limit")
    }
    if ($AnchorTokenCounts) {
        $arguments += @("--anchor-token-counts", $AnchorTokenCounts)
    }
    foreach ($agent in $Disabled) {
        $arguments += @("--disable-unit", $agent)
    }
    Write-Host "[agentic-matrix] $Name"
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment $Name failed with exit code $LASTEXITCODE"
    }
}

# Main comparison table.
if (-not $InitialPredictions) {
    Invoke-AgenticExperiment "main_one_pass" "one_pass" 0 @() "off"
    $InitialPredictions = Join-Path $OutputDir "main_one_pass.jsonl"
} else {
    Invoke-AgenticExperiment "main_one_pass" "one_pass" 0 @() "off"
}
Invoke-AgenticExperiment "main_confidence_gated" "confidence_gated" 1 @() "off"
Invoke-AgenticExperiment "main_parent_only" "parent_only" 1 @() "off"
Invoke-AgenticExperiment "main_static_all" "static_all" 3 @() "off"
Invoke-AgenticExperiment "main_hierarchical" "hierarchical" 3 @() $FeedbackMode

# Component ablations. There is deliberately no Query Rewrite agent.
Invoke-AgenticExperiment "ablation_without_target" "hierarchical" 3 @("target") $FeedbackMode
Invoke-AgenticExperiment "ablation_without_context" "hierarchical" 3 @("context") $FeedbackMode
Invoke-AgenticExperiment "ablation_without_relation" "hierarchical" 3 @("relation") $FeedbackMode
Invoke-AgenticExperiment "ablation_without_zoom" "hierarchical" 3 @("zoom") $FeedbackMode

# Mechanism ablations.
Invoke-AgenticExperiment "ablation_without_constraint_graph" "hierarchical" 3 @() $FeedbackMode @("--no-constraint-graph")
Invoke-AgenticExperiment "ablation_without_semantic_frame_protection" "hierarchical" 3 @() $FeedbackMode @("--no-semantic-frame-protection")
Invoke-AgenticExperiment "ablation_without_false_repair_guard" "hierarchical" 3 @() $FeedbackMode @("--no-false-repair-guard")

# Cost-accuracy curve; Relation Agent is reasoning-only and does not consume a perception call.
foreach ($budget in 0, 1, 2) {
    Invoke-AgenticExperiment "budget_k$budget" "hierarchical" $budget @() $FeedbackMode
}

& $Python $summaryScript `
    "--input-dir" $OutputDir `
    "--csv-output" (Join-Path $OutputDir "agentic_matrix.csv") `
    "--json-output" (Join-Path $OutputDir "agentic_matrix.json")
if ($LASTEXITCODE -ne 0) {
    throw "Matrix summarization failed with exit code $LASTEXITCODE"
}
