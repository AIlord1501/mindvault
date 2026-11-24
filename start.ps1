<#
.SYNOPSIS
    Launch or verify MindVault (local backend on 127.0.0.1:8765 + Electron desktop client).
.PARAMETER Check
    Run run.py --check, backend tests, and frontend checks without launching UI windows.
.PARAMETER BackendOnly
    Run only the local FastAPI backend bound strictly to 127.0.0.1.
.PARAMETER FrontendOnly
    Run only the Electron desktop client (without starting a separate backend here).
#>
[CmdletBinding()]
param(
    [switch]$Check,
    [switch]$BackendOnly,
    [switch]$FrontendOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
Push-Location $RepoRoot

function Resolve-Python {
    $venvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPy) {
        return $venvPy
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        throw "Python executable not found. Activate your virtualenv or install Python 3.9+."
    }
    return $cmd.Source
}

function Resolve-Npm {
    $cmd = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        $cmd = Get-Command npm -ErrorAction SilentlyContinue
    }
    if ($null -eq $cmd) {
        throw "npm executable not found. Install Node.js 18+."
    }
    return $cmd.Source
}

try {
    $PythonExe = Resolve-Python

    if ($Check) {
        Write-Host "==> Validating MindVault configuration & test suites..."
        & $PythonExe (Join-Path $RepoRoot "run.py") --check
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $PythonExe (Join-Path $RepoRoot "scripts\dev_check.py")
        exit $LASTEXITCODE
    }

    if ($BackendOnly) {
        Write-Host "==> Starting MindVault backend strictly on http://127.0.0.1:8765 ..."
        & $PythonExe (Join-Path $RepoRoot "run.py")
        exit $LASTEXITCODE
    }

    $NpmExe = Resolve-Npm
    $FrontendDir = Join-Path $RepoRoot "frontend"

    if ($FrontendOnly) {
        Write-Host "==> Starting MindVault Electron desktop shell..."
        Push-Location $FrontendDir
        try {
            & $NpmExe start
            exit $LASTEXITCODE
        } finally {
            Pop-Location
        }
    }

    Write-Host "==> Starting MindVault local backend on http://127.0.0.1:8765 ..."
    $BackendProc = Start-Process -FilePath $PythonExe `
        -ArgumentList @((Join-Path $RepoRoot "run.py")) `
        -WorkingDirectory $RepoRoot `
        -PassThru `
        -NoNewWindow

    try {
        $Healthy = $false
        for ($i = 0; $i -lt 30; $i++) {
            if ($BackendProc.HasExited) {
                throw "Backend exited prematurely with code $($BackendProc.ExitCode)."
            }
            try {
                $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8765/health" -TimeoutSec 2
                if ($resp.status -eq "ok") {
                    $Healthy = $true
                    break
                }
            } catch {
                Start-Sleep -Milliseconds 500
            }
        }
        if (-not $Healthy) {
            throw "MindVault backend failed health check on http://127.0.0.1:8765/health."
        }

        Write-Host "==> Backend healthy. Launching Electron desktop client..."
        $env:MINDVAULT_SKIP_BACKEND_SPAWN = "1"
        Push-Location $FrontendDir
        try {
            & $NpmExe start
        } finally {
            Pop-Location
        }
    } finally {
        if ($null -ne $BackendProc -and -not $BackendProc.HasExited) {
            Write-Host "==> Stopping MindVault local backend..."
            Stop-Process -Id $BackendProc.Id -Force -ErrorAction SilentlyContinue
        }
    }
} finally {
    Pop-Location
}
