param(
    [Parameter(Mandatory = $true)]
    [string]$RaceDate,

    [Parameter(Mandatory = $true)]
    [string]$PreRaceFeatureCsv,

    [Parameter(Mandatory = $false)]
    [string]$OutputRoot = "outputs\place_market_offset_forward_paper_phase6c_v2",

    [Parameter(Mandatory = $false)]
    [string]$OddsSnapshotType = "FINAL_ODDS",

    [Parameter(Mandatory = $false)]
    [string]$OddsObservedAt = ""
)

$ErrorActionPreference = "Stop"

$inputDir = Join-Path $OutputRoot "input"
New-Item -ItemType Directory -Force -Path $inputDir | Out-Null

$dateStem = $RaceDate.Replace("-", "")
$predictionCsv = Join-Path $inputDir "pre_race_predictions_$dateStem.csv"

$prepareArgs = @(
    "scripts\prepare_place_forward_predictions_phase6c_v2.py",
    "--race-date", $RaceDate,
    "--pre-race-feature-csv", $PreRaceFeatureCsv,
    "--output-csv", $predictionCsv,
    "--odds-snapshot-type", $OddsSnapshotType
)
if ($OddsObservedAt -ne "") {
    $prepareArgs += @("--odds-observed-at", $OddsObservedAt)
}

python @prepareArgs

python scripts\run_place_market_offset_forward_paper_phase6c_v2.py predict `
    --race-date $RaceDate `
    --input-csv $predictionCsv `
    --output-root $OutputRoot
