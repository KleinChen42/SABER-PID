param(
  [string]$RemoteHost = 'h200-codex',
  [string]$RemoteRoot = '/home/hera/pid_reliability_benchmark/outputs/final_replication',
  [string]$LocalRoot = 'outputs/final_replication'
)
$ErrorActionPreference = 'Stop'
$files = @('qwen8_a_p1_768.jsonl','qwen8_a_p1_3072.jsonl','qwen8_a_p2_768.jsonl','qwen8_a_p2_3072.jsonl','qwen8_b_p0_768.jsonl','qwen8_b_p0_3072.jsonl','qwen8_b_p1_768.jsonl','qwen8_b_p1_3072.jsonl','qwen8_b_p2_768.jsonl','qwen8_b_p2_3072.jsonl')
$ssh = Join-Path $env:WINDIR 'System32/OpenSSH/ssh.exe'
$localDir = Join-Path (Get-Location) $LocalRoot
[IO.Directory]::CreateDirectory($localDir) | Out-Null
foreach ($file in $files) {
  $remotePath = "$RemoteRoot/$file"
  $si = New-Object System.Diagnostics.ProcessStartInfo
  $si.FileName = $ssh
  $si.Arguments = "-o PasswordAuthentication=no -o KbdInteractiveAuthentication=no -o PreferredAuthentications=publickey -o ConnectTimeout=15 $RemoteHost `"base64 -w0 $remotePath`""
  $si.UseShellExecute = $false; $si.RedirectStandardOutput = $true; $si.RedirectStandardError = $true
  $p = New-Object System.Diagnostics.Process; $p.StartInfo = $si; [void]$p.Start()
  $encoded = $p.StandardOutput.ReadToEnd(); $err = $p.StandardError.ReadToEnd(); $p.WaitForExit()
  if ($p.ExitCode -ne 0 -or [string]::IsNullOrWhiteSpace($encoded)) { throw "pull failed for $file (exit=$($p.ExitCode)): $err" }
  $bytes = [Convert]::FromBase64String(($encoded -replace '\s',''))
  [IO.File]::WriteAllBytes((Join-Path $localDir $file), $bytes)
  Write-Output ("pulled {0} bytes={1}" -f $file,$bytes.Length)
}
