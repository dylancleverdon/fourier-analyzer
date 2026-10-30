' Starts Fourier Analyzer without a console window. Copied to
' %LOCALAPPDATA%\FourierAnalyzer\Fourier Analyzer.vbs by the installer and
' refreshed by every update; the Start menu and desktop shortcuts run it.
Set sh = CreateObject("WScript.Shell")
home = sh.ExpandEnvironmentStrings("%LOCALAPPDATA%") & "\FourierAnalyzer"
Set env = sh.Environment("Process")
env("PYTHONPATH") = home & "\app"
env("PYTHONUNBUFFERED") = "1"
env("PATH") = home & "\venv\Scripts;" & sh.ExpandEnvironmentStrings("%USERPROFILE%") & "\.local\bin;" & env("PATH")
' run from the install folder, not from app\ (Windows cannot swap a folder
' that is a running program's working directory, which updates need)
sh.CurrentDirectory = home
sh.Run """" & home & "\venv\Scripts\pythonw.exe"" -m fourier_analyzer app", 0, False
