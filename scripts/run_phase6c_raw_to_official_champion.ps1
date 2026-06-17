param(
    [Parameter(Mandatory = $true)]
    [string]$RaceDate,

    [Parameter(Mandatory = $true)]
    [string]$RawPreRaceCsv,

    [Parameter(Mandatory = $false)]
    [string]$OutputRoot = "outputs\place_market_offset_forward_paper_phase6c_v2"
)

$ErrorActionPreference = "Stop"

python scripts\run_phase6c_raw_to_official_champion.py `
    --race-date $RaceDate `
    --raw-pre-race-csv $RawPreRaceCsv `
    --phase6c-output-root $OutputRoot

exit $LASTEXITCODE
