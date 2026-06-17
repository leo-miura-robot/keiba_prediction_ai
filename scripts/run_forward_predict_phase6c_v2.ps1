param(
  [string]$RaceDate,
  [string]$InputCsv = "",
  [string]$OutputRoot = "outputs\place_market_offset_forward_paper_phase6c_v2"
)

$args = @("scripts\run_place_market_offset_forward_paper_phase6c_v2.py", "predict", "--race-date", $RaceDate, "--output-root", $OutputRoot)
if ($InputCsv -ne "") { $args += @("--input-csv", $InputCsv) }
python @args
