# Windows equivalent of `make ci` (make is often missing on PATH).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "== fmt ==" -ForegroundColor Cyan
cargo fmt --all -- --check
if ($LASTEXITCODE -ne 0) { throw "fmt failed" }

Write-Host "== clippy ==" -ForegroundColor Cyan
cargo clippy --workspace --all-targets -- -D warnings
if ($LASTEXITCODE -ne 0) { throw "clippy failed" }

Write-Host "== test ==" -ForegroundColor Cyan
cargo test --workspace --all-targets -- --test-threads=2
if ($LASTEXITCODE -ne 0) { throw "test failed" }

Write-Host "CI OK" -ForegroundColor Green
