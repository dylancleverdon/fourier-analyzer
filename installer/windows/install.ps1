# Fourier Analyzer installer for Windows 10 / 11 (64-bit).
#
#   Double-click "Install-Fourier-Analyzer-Windows.cmd" from the latest release, or in PowerShell:
#   irm https://github.com/dylancleverdon/fourier-analyzer/releases/latest/download/install.ps1 | iex
#
# Everything goes into your user account (no admin rights needed):
#   %LOCALAPPDATA%\FourierAnalyzer\{app,venv,models,data}
#   Start menu + desktop shortcut "Fourier Analyzer"
#
# Environment overrides (optional):
#   FA_CHANNEL=branch:<name>   install from a branch instead of the latest release
#   FA_SOURCE=C:\path\checkout install from a local folder (development)
#   FA_SKIP_MODELS=1           do not pre-download the separation models
#   FA_NO_LAUNCH=1             do not start the app at the end

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'      # Invoke-WebRequest is very slow with its progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Repo = 'dylancleverdon/fourier-analyzer'
$AppName = 'Fourier Analyzer'
$HomeDir = Join-Path $env:LOCALAPPDATA 'FourierAnalyzer'
$PyVersion = '3.12'
$Channel = if ($env:FA_CHANNEL) { $env:FA_CHANNEL } else { 'release' }

function Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Magenta }
function Warn($text) { Write-Host $text -ForegroundColor Yellow }
function Die($text) { Write-Host "ERROR: $text" -ForegroundColor Red; throw $text }

New-Item -ItemType Directory -Force -Path $HomeDir | Out-Null
try { Start-Transcript -Append -Path (Join-Path $HomeDir 'install.log') | Out-Null } catch {}
Write-Host "Installing $AppName (Windows $([Environment]::OSVersion.Version), $env:PROCESSOR_ARCHITECTURE)" -ForegroundColor White
if (-not [Environment]::Is64BitOperatingSystem) { Die "64-bit Windows is required" }

try {
# ---------------------------------------------------------------- 1. uv + python
Step "1/7  Python toolchain (uv)"
$LocalBin = Join-Path $env:USERPROFILE '.local\bin'
$env:Path = "$LocalBin;$env:Path"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    $env:UV_NO_MODIFY_PATH = '1'
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$LocalBin;$env:Path"
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Die "uv installation failed" }
& uv --version

# ---------------------------------------------------------------- 2. app code
Step "2/7  Downloading the app ($Channel)"
$Tmp = Join-Path ([IO.Path]::GetTempPath()) ("fa-install-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path "$Tmp\src" | Out-Null
$Version = ''
if ($env:FA_SOURCE) {
    Write-Host "using local source $env:FA_SOURCE"
    Copy-Item -Recurse -Force (Join-Path $env:FA_SOURCE '*') "$Tmp\src"
} else {
    if ($Channel.StartsWith('branch:')) {
        $Branch = $Channel.Substring(7)
        $Url = "https://codeload.github.com/$Repo/tar.gz/refs/heads/$Branch"
    } else {
        try {
            $Rel = Invoke-RestMethod -Headers @{ 'Accept' = 'application/vnd.github+json'; 'User-Agent' = 'fa-installer' } `
                "https://api.github.com/repos/$Repo/releases/latest"
            $Version = $Rel.tag_name.TrimStart('v')
        } catch { $Version = '' }
        if ($Version) {
            $Url = "https://github.com/$Repo/releases/download/v$Version/fourier-analyzer.tar.gz"
        } else {
            Warn "no GitHub release found yet; installing the default branch"
            $Url = "https://codeload.github.com/$Repo/tar.gz/HEAD"
        }
    }
    Write-Host "downloading $Url"
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile "$Tmp\app.tar.gz"
    & tar.exe -xzf "$Tmp\app.tar.gz" -C "$Tmp\src"
    if ($LASTEXITCODE -ne 0) { Die "could not unpack the download" }
    if (-not (Test-Path "$Tmp\src\fourier_analyzer")) {
        # GitHub source archives have one top-level folder
        $Inner = Get-ChildItem -Directory "$Tmp\src" | Select-Object -First 1
        Move-Item $Inner.FullName "$Tmp\src2"
        Remove-Item -Recurse -Force "$Tmp\src"
        Move-Item "$Tmp\src2" "$Tmp\src"
    }
    if ($Version) { Set-Content -Path "$Tmp\src\VERSION" -Value $Version -Encoding ascii }
}
if (-not (Test-Path "$Tmp\src\fourier_analyzer")) { Die "downloaded archive does not contain the app" }
$Version = (Get-Content "$Tmp\src\VERSION" -Raw).Trim()
# stop a running copy so its files can be replaced
Get-CimInstance Win32_Process -Filter "Name like 'python%'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*FourierAnalyzer*" -or $_.ExecutablePath -like "$HomeDir*" } |
    ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force } catch {} }
if (Test-Path "$HomeDir\app.old") { Remove-Item -Recurse -Force "$HomeDir\app.old" }
if (Test-Path "$HomeDir\app") { Move-Item "$HomeDir\app" "$HomeDir\app.old" }
Move-Item "$Tmp\src" "$HomeDir\app"
if (Test-Path "$HomeDir\app.old") { Remove-Item -Recurse -Force "$HomeDir\app.old" -ErrorAction SilentlyContinue }
Write-Host "app version $Version"

# ---------------------------------------------------------------- 3. venv + deps
Step "3/7  Python $PyVersion environment + dependencies (this takes a few minutes the first time)"
$Py = "$HomeDir\venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { & uv venv --seed --python $PyVersion "$HomeDir\venv"; if ($LASTEXITCODE -ne 0) { Die "could not create the Python environment" } }
& uv pip install --python $Py -r "$HomeDir\app\requirements.txt"
if ($LASTEXITCODE -ne 0) { Die "installing dependencies failed (see the messages above)" }
& uv pip install --python $Py imageio-ffmpeg
& $Py -c "import onnxruntime, audioread; from audio_separator.separator import Separator; print('separation stack OK')"
if ($LASTEXITCODE -ne 0) {
    Warn "separation stack incomplete; retrying install"
    & uv pip install --python $Py --reinstall -r "$HomeDir\app\requirements.txt"
}
$Ffmpeg = (& $Py -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())").Trim()
Copy-Item -Force $Ffmpeg "$HomeDir\venv\Scripts\ffmpeg.exe"
Write-Host "ffmpeg: $Ffmpeg"

# ---------------------------------------------------------------- 4. launcher + shortcuts
Step "4/7  Creating the $AppName shortcuts"
Copy-Item -Force "$HomeDir\app\installer\windows\launch.vbs" "$HomeDir\Fourier Analyzer.vbs"
Copy-Item -Force "$HomeDir\app\installer\windows\uninstall.ps1" "$HomeDir\uninstall.ps1"
$Icon = "$HomeDir\FourierAnalyzer.ico"
try { & $Py "$HomeDir\app\installer\make_icon.py" $Icon } catch { Warn "no icon: $_" }
$Shell = New-Object -ComObject WScript.Shell
$StartMenu = Join-Path ([Environment]::GetFolderPath('Programs')) $AppName
New-Item -ItemType Directory -Force -Path $StartMenu | Out-Null
function New-Shortcut($Path, $Target, $Arguments, $Description) {
    $s = $Shell.CreateShortcut($Path)
    $s.TargetPath = $Target
    $s.Arguments = $Arguments
    $s.WorkingDirectory = $HomeDir
    if (Test-Path $Icon) { $s.IconLocation = $Icon }
    $s.Description = $Description
    $s.Save()
}
$Wscript = Join-Path $env:WINDIR 'System32\wscript.exe'
$Launch = "`"$HomeDir\Fourier Analyzer.vbs`""
New-Shortcut (Join-Path $StartMenu "$AppName.lnk") $Wscript $Launch 'Separate, analyse and rebuild sounds as Serum 2 presets'
New-Shortcut (Join-Path ([Environment]::GetFolderPath('Desktop')) "$AppName.lnk") $Wscript $Launch 'Fourier Analyzer'
New-Shortcut (Join-Path $StartMenu "Uninstall $AppName.lnk") 'powershell.exe' `
    "-NoProfile -ExecutionPolicy Bypass -File `"$HomeDir\uninstall.ps1`"" "Remove $AppName"
Write-Host "Start menu: $StartMenu"

# ---------------------------------------------------------------- 5. models
Step "5/7  Downloading separation models (Demucs + DrumSep, ~1 GB)"
$env:PYTHONPATH = "$HomeDir\app"
Push-Location $HomeDir
if (-not $env:FA_SKIP_MODELS) {
    & $Py -m fourier_analyzer models
    if ($LASTEXITCODE -ne 0) { Warn "model download failed; the app will retry on first use" }
}

# ---------------------------------------------------------------- 6. Serum detection
Step "6/7  Looking for Serum 2"
$Detect = @'
from fourier_analyzer import config
print("Serum 2 plugin :", config.serum_plugin_path() or "NOT FOUND (checks in real Serum disabled; set the path in Settings)")
print("Serum tables   :", config.serum_tables_dir() or "NOT FOUND (set it in Settings)")
print("Serum presets  :", config.serum_presets_dir() or "NOT FOUND")
config.save_settings(update_channel=config.settings().get("update_channel", "release"),
                     auto_update=config.settings().get("auto_update", True))
'@
$Detect | & $Py -
if ($Channel.StartsWith('branch:')) {
    & $Py -c "from fourier_analyzer import config; config.save_settings(update_channel='$Channel')"
}
Pop-Location

# ---------------------------------------------------------------- 7. done
Step "7/7  Done"
Write-Host "$AppName $Version installed." -ForegroundColor Green
Write-Host "  App:      Start menu / desktop -> $AppName"
Write-Host "  Exports:  $env:USERPROFILE\Music\Fourier Analyzer"
Write-Host "  Presets:  Serum 2 browser -> User -> Fourier Analyzer"
Write-Host "  Updates:  automatic on launch (Settings to change)"
Write-Host "  Log:      $HomeDir\install.log"
if (-not $env:FA_NO_LAUNCH) { Start-Process $Wscript -ArgumentList $Launch -WorkingDirectory $HomeDir }
} finally {
    if ($Tmp -and (Test-Path $Tmp)) { Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue }
    try { Stop-Transcript | Out-Null } catch {}
}
