# Removes Fourier Analyzer (the app, its Python environment, models and shortcuts).
# Your exports in Music\Fourier Analyzer and presets installed into Serum 2 are kept.
$AppName = 'Fourier Analyzer'
$HomeDir = Join-Path $env:LOCALAPPDATA 'FourierAnalyzer'
$answer = Read-Host "Remove $AppName from this PC? Your projects in $HomeDir\data will be deleted too. [y/N]"
if ($answer -notmatch '^[yY]') { Write-Host "Nothing removed."; exit 0 }
Get-CimInstance Win32_Process -Filter "Name like 'python%'" -ErrorAction SilentlyContinue |
    Where-Object { $_.ExecutablePath -like "$HomeDir*" } |
    ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force } catch {} }
Start-Sleep -Seconds 1
Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path ([Environment]::GetFolderPath('Desktop')) "$AppName.lnk")
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path ([Environment]::GetFolderPath('Programs')) $AppName)
Set-Location $env:TEMP
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $HomeDir
Write-Host "$AppName removed. Exports in $env:USERPROFILE\Music\Fourier Analyzer were kept."
Read-Host "Press Enter to close"
