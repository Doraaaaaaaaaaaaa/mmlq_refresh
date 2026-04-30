param(
    [string]$Stamp = (Get-Date -Format "yyyyMMdd_HHmmss"),
    [string]$Python = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = $PSScriptRoot
Set-Location -LiteralPath $Root

if (-not $Python) {
    $pythonCandidates = @()
    if ($env:ABLATION_PYTHON) {
        $pythonCandidates += $env:ABLATION_PYTHON
    }
    if ($env:CONDA_PREFIX) {
        $pythonCandidates += (Join-Path $env:CONDA_PREFIX "python.exe")
    }
    if ($env:USERPROFILE) {
        $pythonCandidates += (Join-Path $env:USERPROFILE "anaconda\envs\ammnet-gpu\python.exe")
        $pythonCandidates += (Join-Path $env:USERPROFILE "anaconda3\envs\ammnet-gpu\python.exe")
        $pythonCandidates += (Join-Path $env:USERPROFILE "miniconda3\envs\ammnet-gpu\python.exe")
    }
    $Python = $pythonCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
}

if (-not $Python) {
    throw "Could not find a Python executable. Pass -Python or set ABLATION_PYTHON to the ammnet-gpu python.exe path."
}
if (Test-Path -LiteralPath $Python) {
    $Python = (Resolve-Path -LiteralPath $Python).Path
}

$SummaryLog = Join-Path $Root ("ablation_remaining_{0}.summary.log" -f $Stamp)

$Runs = @(
    @{ Id = "A2"; Mode = "concat";    Desc = "simple image-text concat" },
    @{ Id = "A3"; Mode = "direct_ca"; Desc = "MMLQ-style direct learnable-query cross-attention" },
    @{ Id = "A4"; Mode = "icif";      Desc = "ICIF without hierarchical attention" },
    @{ Id = "A5"; Mode = "icif_ha";   Desc = "ICIF with hierarchical attention, without attributes" },
    @{ Id = "A6"; Mode = $null;       Desc = "full model with attribute reasoning" }
)

"[{0}] starting remaining ablations: A2 A3 A4 A5 A6" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss") |
    Add-Content -LiteralPath $SummaryLog

foreach ($Run in $Runs) {
    $runKey = $Run.Id.ToLower()
    $outLog = Join-Path $Root ("ablation_{0}_{1}.out.log" -f $runKey, $Stamp)
    $errLog = Join-Path $Root ("ablation_{0}_{1}.err.log" -f $runKey, $Stamp)

    "-----" | Add-Content -LiteralPath $SummaryLog
    "[{0}] {1}: {2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Run.Id, $Run.Desc |
        Add-Content -LiteralPath $SummaryLog
    "stdout: {0}" -f $outLog | Add-Content -LiteralPath $SummaryLog
    "stderr: {0}" -f $errLog | Add-Content -LiteralPath $SummaryLog

    $args = @("-u", "main.py", "--config", "config.yml")
    if ($null -ne $Run.Mode) {
        $args += @("--ablation-mode", $Run.Mode)
    }

    if ($DryRun) {
        "dry_run: {0} {1}" -f $Python, ($args -join " ") |
            Add-Content -LiteralPath $SummaryLog
        continue
    }

    & $Python @args 1>> $outLog 2>> $errLog
    $exitCode = $LASTEXITCODE

    "[{0}] {1} exit_code={2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Run.Id, $exitCode |
        Add-Content -LiteralPath $SummaryLog

    if ($exitCode -ne 0) {
        exit $exitCode
    }
}

"[{0}] all remaining ablations finished" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss") |
    Add-Content -LiteralPath $SummaryLog
