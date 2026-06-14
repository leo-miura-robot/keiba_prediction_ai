# Resume Design V2.1.1

Strict resume validates contiguous completion from 2016, state pickle, output parquet, row count, state version, completed year, input fingerprint, feature set hash, and code bundle hash. Git SHA is recorded but not used as a resume-failure condition.

Use `--rebuild-from-year YYYY` to regenerate YYYY and later. For YYYY > 2016, the previous year state is required even if the previous year is not included in `--years`.
