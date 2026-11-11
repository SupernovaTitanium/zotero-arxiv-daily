# Daily digest dispatcher. Runs the pipeline on Light1, else Light2, else this
# PC. A remote run gets a fresh copy of the code, config and state, receives the
# credentials on SSH stdin (never written to the remote disk), and its state and
# outputs are copied back after it succeeds.
#   -Check            validate config and credentials, probe every host; no run
#   -TestHost Light1  real run (sends an email) on that host without the dedup
#                     history; nothing is copied back
# Keep this file ASCII: Windows PowerShell 5.1 reads BOM-less scripts as ANSI.
param(
    [switch]$Check,
    [ValidateSet('Light1', 'Light2')][string]$TestHost
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$localPython = 'C:\Users\light\venvs\zotero-arxiv-daily-cuda\Scripts\python.exe'
$envFile = Join-Path $repo 'config\local.env.json'
$marker = Join-Path $repo 'state\remote-run.json'
$ssh = Join-Path $env:SystemRoot 'System32\OpenSSH\ssh.exe'
$scp = Join-Path $env:SystemRoot 'System32\OpenSSH\scp.exe'
$tar = Join-Path $env:SystemRoot 'System32\tar.exe'  # Git's tar misreads C: paths
$remoteDir = 'D:\zotero-arxiv-daily'
$remotePython = $remoteDir + '\.venv\Scripts\python.exe'
$remoteTar = 'C:\Windows\System32\tar.exe'
$remoteUv = 'C:\Users\light\.local\bin\uv.exe'
$hosts = [ordered]@{
    # Light1's CPU is shared with other offloaded jobs (CPU embedding measured at 5 s per paper),
    # its RTX 4060 is not: CUDA PyTorch replaces the lockfile's CPU build there.
    Light1 = @{ Target = 'light@100.123.129.82'; Key = 'C:/Users/light/.ssh/id_ed25519_light1'; Torch = 'cu128' }
    Light2 = @{ Target = 'light@100.123.137.57'; Key = 'C:/Users/light/.ssh/id_ed25519-remote-ssh'; Torch = $null }
}
$lock = $null
$previousEnv = @{}
$secretValues = @()
$exitCode = 1
$logDir = Join-Path $repo 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logPrefix = if ($TestHost) { 'testrun' } else { 'digest' }
$log = Join-Path $logDir ('{0}-{1}-{2}.log' -f $logPrefix, (Get-Date -Format 'yyyyMMdd-HHmmss-fff'), $PID)
# Python runs with -X utf8 here and remotely; decode its output the same way.
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false

function Write-Log($message) {
    $text = [string]$message
    foreach ($value in $secretValues) {
        $text = $text.Replace($value, '[REDACTED]')
    }
    ('{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $text) |
        Add-Content -LiteralPath $log -Encoding UTF8
}

# All native calls go through cmd so stderr is merged before PowerShell sees it
# (Windows PowerShell turns native stderr into terminating errors under Stop)
# and so a file can be redirected to stdin byte for byte.
function Invoke-Cmd([string]$line) {
    # Passed through a variable: Windows PowerShell mangles embedded quotes in
    # native arguments, while cmd expands the variable before parsing it.
    $env:ZAD_COMMAND_LINE = $line
    & $env:ComSpec /d /s /c "%ZAD_COMMAND_LINE% 2>&1"
}

function Get-SshOptions($name) {
    '-o BatchMode=yes -o ConnectTimeout=15 -o IdentitiesOnly=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -i {0}' -f $hosts[$name].Key
}

# Remote commands run under cmd.exe; keep them free of double quotes.
function Invoke-Ssh($name, [string]$remoteCommand, [string]$stdinFile) {
    if ($stdinFile) {
        Invoke-Cmd ('"{0}" {1} {2} "{3}" < "{4}"' -f $ssh, (Get-SshOptions $name), $hosts[$name].Target, $remoteCommand, $stdinFile)
    } else {
        Invoke-Cmd ('"{0}" -n {1} {2} "{3}"' -f $ssh, (Get-SshOptions $name), $hosts[$name].Target, $remoteCommand)
    }
}

function Invoke-Scp($name, [string]$source, [string]$destination) {
    Invoke-Cmd ('"{0}" -q {1} "{2}" "{3}"' -f $scp, (Get-SshOptions $name), $source, $destination)
}

$probeScript = @'
$ProgressPreference = 'SilentlyContinue'
$health = 0
try { $health = (Invoke-WebRequest -Uri http://127.0.0.1:10100/health -UseBasicParsing -TimeoutSec 5).StatusCode } catch {}
$load = [math]::Round((1..3 | ForEach-Object { Start-Sleep -Milliseconds 700; (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average } | Measure-Object -Average).Average)
$free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
@{ computer = $env:COMPUTERNAME; health = $health; load = $load; free_gb = $free; uv = (Test-Path 'C:\Users\light\.local\bin\uv.exe') } | ConvertTo-Json -Compress
'@
$probeCommand = 'powershell -NoProfile -NonInteractive -EncodedCommand ' +
    [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($probeScript))

# Returns $null when the host can take the run, else the reason it cannot.
function Test-RemoteHost($name) {
    for ($attempt = 1; $attempt -le 2; $attempt++) {
        $output = @(Invoke-Ssh $name $probeCommand)
        if ($LASTEXITCODE -ne 255) { break }
        if ($attempt -eq 1) { Start-Sleep -Seconds 10 }
    }
    if ($LASTEXITCODE -ne 0) {
        return ('unreachable (ssh exit {0}: {1})' -f $LASTEXITCODE, (($output | Select-Object -Last 1) -as [string]))
    }
    $json = $output | Where-Object { ([string]$_).StartsWith('{') } | Select-Object -Last 1
    if (-not $json) { return 'probe returned no status' }
    $s = $json | ConvertFrom-Json
    $summary = '{0}: proxy health {1}, CPU load {2}%, free RAM {3} GB' -f $s.computer, $s.health, $s.load, $s.free_gb
    if ($s.health -ne 200) { return "local model proxy is down ($summary)" }
    if (-not $s.uv) { return "uv is missing ($summary)" }
    if ($s.load -ge 90) { return "busy ($summary)" }
    if ($s.free_gb -lt 3) { return "low on memory ($summary)" }
    Write-Log ('{0} available ({1})' -f $name, $summary)
    return $null
}

# Copies code, config and state to the remote work directory and syncs its
# environment. Nothing has run yet, so a failure here can fall through to the
# next host.
function Send-Workspace($name, [bool]$withHistory) {
    $files = @('src', 'scripts/remote_entry.py', 'config/base.yaml', 'config/custom.yaml',
        'pyproject.toml', 'uv.lock', '.python-version')
    if ($withHistory -and (Test-Path -LiteralPath (Join-Path $repo 'state\recommended.json'))) {
        $files += 'state/recommended.json'
    }
    foreach ($optional in 'state/corpus_embeddings.npz', 'preferences.yaml') {
        if (Test-Path -LiteralPath (Join-Path $repo $optional)) { $files += $optional }
    }
    $pushTar = Join-Path $env:TEMP ('zotero-arxiv-push-{0}.tar' -f $PID)
    try {
        Invoke-Cmd ('"{0}" -cf "{1}" -C "{2}" --exclude __pycache__ {3}' -f $tar, $pushTar, $repo, ($files -join ' ')) |
            ForEach-Object { Write-Log "  $_" }
        if ($LASTEXITCODE -ne 0) { throw "packing failed (exit $LASTEXITCODE)" }
        Invoke-Ssh $name ('if not exist {0} mkdir {0}' -f $remoteDir) | ForEach-Object { Write-Log "  $_" }
        Invoke-Scp $name $pushTar ('{0}:D:/zotero-arxiv-daily/push.tar' -f $hosts[$name].Target) |
            ForEach-Object { Write-Log "  $_" }
        if ($LASTEXITCODE -ne 0) { throw "copy failed (exit $LASTEXITCODE)" }
    } finally {
        Remove-Item -LiteralPath $pushTar -Force -ErrorAction SilentlyContinue
    }
    # The environment is re-synced only when uv.lock changed since the last sync,
    # so the CUDA build on Light1 is not swapped back and forth every run.
    # After changing a host's Torch value, delete its .venv.
    $sync = '{0} sync --frozen --no-dev --no-install-project --quiet' -f $remoteUv
    if ($hosts[$name].Torch) {
        $sync += (' && {0} pip install --python .venv\Scripts\python.exe --quiet --reinstall-package torch ' +
            '--extra-index-url https://download.pytorch.org/whl/{1} torch==2.11.0+{1}') -f $remoteUv, $hosts[$name].Torch
    }
    $prepare = ('cd /d {0} && (if exist src rmdir /s /q src) && (if exist state rmdir /s /q state) && ' +
        '(if exist output rmdir /s /q output) && {1} -xf push.tar && del push.tar && ' +
        '(fc /b uv.lock .venv\synced-uv.lock >nul 2>&1 || ({2} && copy /y uv.lock .venv\synced-uv.lock >nul))') -f $remoteDir, $remoteTar, $sync
    Invoke-Ssh $name $prepare | ForEach-Object { Write-Log "  $_" }
    if ($LASTEXITCODE -ne 0) { throw "unpacking or environment sync failed (exit $LASTEXITCODE)" }
}

function Install-File([string]$source, [string]$destination) {
    if (Test-Path -LiteralPath $destination) {
        # A plain $null reaches .NET as "" ("The path is not of a legal form.").
        [System.IO.File]::Replace($source, $destination, [System.Management.Automation.Language.NullString]::Value)
    } else {
        Move-Item -LiteralPath $source -Destination $destination
    }
}

# Copies the remote state (dedup history, embedding cache) and outputs back.
# The remote copy started from ours, so it is never older than ours.
function Receive-Workspace($name) {
    Invoke-Ssh $name ('cd /d {0} && {1} -cf pull.tar state output' -f $remoteDir, $remoteTar) |
        ForEach-Object { Write-Log "  $_" }
    if ($LASTEXITCODE -ne 0) { throw "packing remote state failed (exit $LASTEXITCODE)" }
    $staging = Join-Path $repo ('state\pull-{0}' -f $PID)
    New-Item -ItemType Directory -Path $staging -Force | Out-Null
    try {
        $pullTar = Join-Path $staging 'pull.tar'
        Invoke-Scp $name ('{0}:D:/zotero-arxiv-daily/pull.tar' -f $hosts[$name].Target) $pullTar |
            ForEach-Object { Write-Log "  $_" }
        if ($LASTEXITCODE -ne 0) { throw "copying remote state failed (exit $LASTEXITCODE)" }
        Invoke-Cmd ('"{0}" -xf "{1}" -C "{2}"' -f $tar, $pullTar, $staging) | ForEach-Object { Write-Log "  $_" }
        if ($LASTEXITCODE -ne 0) { throw "unpacking remote state failed (exit $LASTEXITCODE)" }
        foreach ($file in Get-ChildItem -LiteralPath (Join-Path $staging 'state') -File) {
            Install-File $file.FullName (Join-Path $repo ('state\' + $file.Name))
        }
        $outputDir = Join-Path $repo 'output'
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
        foreach ($file in Get-ChildItem -LiteralPath (Join-Path $staging 'output') -File -ErrorAction SilentlyContinue) {
            Copy-Item -LiteralPath $file.FullName -Destination $outputDir -Force
        }
    } finally {
        Remove-Item -LiteralPath $staging -Recurse -Force
    }
}

Push-Location $repo
try {
    if (-not (Test-Path -LiteralPath 'config\custom.yaml' -PathType Leaf)) {
        throw 'Local config/custom.yaml is required.'
    }
    New-Item -ItemType Directory -Path 'state' -Force | Out-Null
    try {
        $lock = [System.IO.File]::Open(
            (Join-Path $repo 'state\local-run.lock'),
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    } catch [System.IO.IOException] {
        throw 'Cannot acquire local run lock; another run may be active.'
    }

    # Never print JSON/parser exceptions: they can contain credential values.
    try {
        $mapping = Get-Content -LiteralPath $envFile -Raw -Encoding UTF8 | ConvertFrom-Json
        $allowed = @('ZOTERO_ID', 'ZOTERO_KEY', 'SENDER', 'RECEIVER',
            'SENDER_PASSWORD', 'OPENAI_API_KEY', 'EMAIL_SMTP_SERVER', 'EMAIL_SMTP_PORT')
        if ($mapping -isnot [System.Management.Automation.PSCustomObject]) {
            throw 'Expected an object.'
        }
        foreach ($entry in $mapping.PSObject.Properties) {
            if ($allowed -cnotcontains $entry.Name -or $entry.Value -isnot [string] -or
                [string]::IsNullOrWhiteSpace($entry.Value)) {
                throw 'Invalid environment entry.'
            }
        }
        foreach ($name in $allowed) {
            if ($mapping.PSObject.Properties.Name -cnotcontains $name) {
                throw 'Missing environment entry.'
            }
        }
    } catch {
        throw 'Invalid or missing config/local.env.json; use the example and enter credentials locally.'
    }
    $secretValues = @('ZOTERO_KEY', 'SENDER_PASSWORD', 'OPENAI_API_KEY') |
        ForEach-Object { [string]$mapping.$_ }

    if ($TestHost -and (Test-Path -LiteralPath $marker)) {
        throw 'An interrupted remote run has not been recovered yet; run the daily job first.'
    }
    if (-not $Check -and -not $TestHost -and (Test-Path -LiteralPath $marker)) {
        try {
            $previous = (Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json).host
            Write-Log ('recovering state from the interrupted run on {0}' -f $previous)
            Receive-Workspace $previous
            Remove-Item -LiteralPath $marker
            Write-Log 'recovered'
        } catch {
            Write-Log ('WARNING: recovery failed, continuing with the local state: ' + $_.Exception.Message)
        }
    }

    $candidates = if ($TestHost) { @($TestHost) } else { @($hosts.Keys) }
    $chosen = $null
    foreach ($name in $candidates) {
        $reason = Test-RemoteHost $name
        if ($reason) {
            Write-Log ('{0} skipped: {1}' -f $name, $reason)
            continue
        }
        if ($Check) {
            if (-not $chosen) { $chosen = $name }
            continue
        }
        try {
            Send-Workspace $name (-not $TestHost)
            $chosen = $name
            break
        } catch {
            Write-Log ('{0} skipped: {1}' -f $name, $_.Exception.Message)
        }
    }
    if ($TestHost -and -not $chosen) { throw "$TestHost is not available for a test run." }

    $localReason = $null
    if ($Check -or -not $chosen) {
        try {
            $health = Invoke-WebRequest -Uri 'http://127.0.0.1:10100/health' -UseBasicParsing -TimeoutSec 5
            if ($health.StatusCode -ne 200) { throw 'Unhealthy proxy.' }
        } catch {
            $localReason = 'local model proxy health check failed at 127.0.0.1:10100'
        }
        if (-not (Test-Path -LiteralPath $localPython -PathType Leaf)) {
            $localReason = 'CUDA virtual environment Python is missing'
        }
    }

    if ($Check) {
        Write-Log $(if ($localReason) { "this PC skipped: $localReason" } else { 'this PC available as fallback' })
        if (-not $chosen -and -not $localReason) { $chosen = 'this PC' }
        if (-not $chosen) { throw 'No host can take the run.' }
        Write-Log ('preflight passed; would run on {0}; no pipeline/model/send executed' -f $chosen)
        $exitCode = 0
    } elseif ($chosen) {
        if (-not $TestHost) {
            @{ host = $chosen; started = (Get-Date -Format 'o') } | ConvertTo-Json |
                Set-Content -LiteralPath $marker -Encoding ASCII
        }
        Write-Log ('digest run started on {0} (gpt-6.1-terra, reasoning_effort=low){1}' -f $chosen,
            $(if ($TestHost) { ', test run without dedup history' } else { '' }))
        Invoke-Ssh $chosen ('{0} -X utf8 -u {1}\scripts\remote_entry.py' -f $remotePython, $remoteDir) $envFile |
            ForEach-Object { Write-Log $_ }
        $exitCode = $LASTEXITCODE
        if ($exitCode -eq 0 -and -not $TestHost) {
            # The email is out; a failed copy-back is retried by the next run.
            try {
                Receive-Workspace $chosen
                Remove-Item -LiteralPath $marker
                Write-Log ('state and outputs copied back from {0}' -f $chosen)
            } catch {
                Write-Log ('Copy-back failed, the next run will retry it: ' + $_.Exception.Message)
                $exitCode = 3
            }
        } elseif ($exitCode -ne 0) {
            Write-Log ('remote run on {0} failed; not retried elsewhere, since it may already have sent an email' -f $chosen)
        }
    } else {
        if ($localReason) { throw "No host can take the run; this PC: $localReason." }
        foreach ($entry in $mapping.PSObject.Properties) {
            $previousEnv[$entry.Name] = [Environment]::GetEnvironmentVariable($entry.Name, 'Process')
            [Environment]::SetEnvironmentVariable($entry.Name, $entry.Value, 'Process')
        }
        # The CUDA environment holds dependencies only; the project itself runs from src.
        $previousEnv['PYTHONPATH'] = [Environment]::GetEnvironmentVariable('PYTHONPATH', 'Process')
        [Environment]::SetEnvironmentVariable('PYTHONPATH', (Join-Path $repo 'src'), 'Process')
        Write-Log 'no remote host available; digest run started on this PC (gpt-6.1-terra, reasoning_effort=low)'
        Invoke-Cmd ('"{0}" -X utf8 -u -m zotero_arxiv_daily.main' -f $localPython) | ForEach-Object { Write-Log $_ }
        $exitCode = $LASTEXITCODE
    }
} catch {
    Write-Log ('Runner failed: ' + $_.Exception.Message)
} finally {
    foreach ($name in $previousEnv.Keys) {
        [Environment]::SetEnvironmentVariable($name, $previousEnv[$name], 'Process')
    }
    if ($null -ne $lock) { $lock.Dispose() }
    Pop-Location
    Write-Log ('runner finished, exit={0}' -f $exitCode)
}
exit $exitCode
