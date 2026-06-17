param(
  [string]$RaceDate,
  [string]$SettlementCsv = "",
  [string]$OutputRoot = "outputs\place_market_offset_forward_paper_phase6c_v2"
)

$args = @("scripts\run_place_market_offset_forward_paper_phase6c_v2.py", "settle", "--race-date", $RaceDate, "--output-root", $OutputRoot)
if ($SettlementCsv -ne "") { $args += @("--settlement-csv", $SettlementCsv) }
python @args
