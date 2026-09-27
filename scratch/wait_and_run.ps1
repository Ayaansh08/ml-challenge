while (!(Test-Path "outputs/final_submission_downstream/features/test_features.parquet")) {
    Start-Sleep -Seconds 5
}
Write-Host "test_features.parquet found!"
& .\scratch\run_rest.ps1
