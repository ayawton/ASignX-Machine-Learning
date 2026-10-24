$ErrorActionPreference = 'Stop'
$auditDirectory = $PSScriptRoot
$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $auditDirectory '../../..')).ProviderPath
$plan = Get-Content -LiteralPath (Join-Path $auditDirectory 'execution-plan.json') -Raw | ConvertFrom-Json
if ($plan.workspace -ne $workspaceRoot) { throw 'Workspace differs from audited workspace' }
$workspacePrefix = $workspaceRoot.TrimEnd('\') + '\'
$targets = @()
foreach ($entry in $plan.delete) {
    $target = (Resolve-Path -LiteralPath (Join-Path $workspaceRoot $entry.path)).ProviderPath
    if (-not $target.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $target" }
    $item = Get-Item -LiteralPath $target -Force
    if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw "Not a regular audited file: $target" }
    if ($plan.protected_sha256.PSObject.Properties.Name -contains $entry.path) { throw "Protected file selected: $target" }
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw "Candidate changed since verification: $target" }
    $targets += $target
}
foreach ($entry in $plan.protected_sha256.PSObject.Properties) {
    $path = Join-Path $workspaceRoot $entry.Name
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) { throw "Protected file changed before deletion: $path" }
}
Write-Output "Validated $($targets.Count) deletion targets and $($plan.protected_sha256.PSObject.Properties.Name.Count) protected files."
foreach ($target in $targets) { Remove-Item -LiteralPath $target -Force }
$verificationFailures = @()
foreach ($entry in $plan.protected_sha256.PSObject.Properties) {
    $path = Join-Path $workspaceRoot $entry.Name
    if (-not (Test-Path -LiteralPath $path)) { $verificationFailures += $entry.Name; continue }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) { $verificationFailures += $entry.Name }
}
$remainingTargets = @($targets | Where-Object { Test-Path -LiteralPath $_ })
$result = [ordered]@{
    completed_utc = [DateTime]::UtcNow.ToString('o')
    deleted_files = $targets.Count
    freed_bytes = [long](($plan.delete | Measure-Object -Property bytes -Sum).Sum)
    protected_files_checked = $plan.protected_sha256.PSObject.Properties.Name.Count
    protected_files_unchanged = ($verificationFailures.Count -eq 0)
    verification_failures = $verificationFailures
    deleted_targets_remaining = $remainingTargets
    skipped_files = $plan.skipped
    dataset_restoration_manifest = $plan.restore_manifest
    git_and_lfs_untouched = $true
}
$result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $auditDirectory 'cleanup-result.json') -Encoding UTF8
[pscustomobject]$result | Select-Object completed_utc,deleted_files,freed_bytes,protected_files_checked,protected_files_unchanged,git_and_lfs_untouched | ConvertTo-Json
if ($verificationFailures.Count -or $remainingTargets.Count) { throw 'Cleanup verification failed; see cleanup-result.json' }
