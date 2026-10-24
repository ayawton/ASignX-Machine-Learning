$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath '.venv-py314\Scripts\python.exe')) {
        & py -3.14 -m venv .venv-py314
        if ($LASTEXITCODE -ne 0) { throw 'Install standard Windows x64 Python 3.14 first.' }
    }
    $projectPython = Join-Path $PSScriptRoot '.venv-py314\Scripts\python.exe'
    & $projectPython -c "import sys; assert sys.version_info[:2] == (3, 14), sys.version"
    if ($LASTEXITCODE -ne 0) { throw 'The project environment must use Python 3.14.' }
    & $projectPython -m pip install -r requirements-tools.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & $projectPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
    Write-Host 'Ready. Use .\.venv-py314\Scripts\python.exe for project commands.'
} finally {
    Pop-Location
}
