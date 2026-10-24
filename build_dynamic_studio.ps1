param([string]$Python = "$PSScriptRoot\.venv-py314\Scripts\python.exe")
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    & $Python -c "import sys; assert sys.version_info[:2] == (3, 14), sys.version"
    if ($LASTEXITCODE -ne 0) { throw 'Build with Python 3.14; run setup_python314.ps1 first.' }
    & $Python -m PyInstaller --noconfirm --workpath build/dynamic-studio ASignX_DynamicStudio.spec
    if ($LASTEXITCODE -ne 0) { throw 'Standalone app build failed.' }
    Copy-Item -LiteralPath docs/dynamic-studio-guide.md -Destination dist/ASignX_DynamicStudio_GUIDE.md
    Write-Host 'Built dist\ASignX_DynamicStudio.exe. Copy this single EXE to another Windows x64 PC.'
} finally {
    Pop-Location
}
