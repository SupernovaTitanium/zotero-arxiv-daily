Option Explicit
Dim shell, fso, runner, command, code
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
runner = fso.BuildPath(fso.GetParentFolderName(WScript.ScriptFullName), "run_digest.ps1")
command = "powershell.exe -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File """ & runner & """"
If WScript.Arguments.Count > 0 Then
    If WScript.Arguments(0) <> "-Check" Then WScript.Quit 1
    command = command & " -Check"
End If
code = shell.Run(command, 0, True)
WScript.Quit code
