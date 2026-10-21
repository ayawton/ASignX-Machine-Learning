param([string]$Python = "$PSScriptRoot\.venv-py314\Scripts\python.exe")
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
$previousPythonPath = $env:PYTHONPATH
try {
    # Also support a project-local packaging installation without changing the venv.
    $localBuildTools = Join-Path $PSScriptRoot 'tmp\dynamic-build-tools'
    if (Test-Path -LiteralPath $localBuildTools) {
        $env:PYTHONPATH = "$localBuildTools;$PSScriptRoot;$previousPythonPath"
    }
    & $Python -m PyInstaller --noconfirm --workpath build/dynamic-studio ASignX_DynamicStudio.spec
    if ($LASTEXITCODE -ne 0) { throw 'Standalone app build failed.' }
    Copy-Item -LiteralPath docs/dynamic-studio-guide.md -Destination dist/ASignX_DynamicStudio_GUIDE.md
    Write-Host 'Built dist\ASignX_DynamicStudio.exe. Copy this single EXE to another Windows x64 PC.'
} finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
