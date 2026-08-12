[CmdletBinding(DefaultParameterSetName = 'shell')]
param(
  [string]$SshHost = 'h200-codex',
  [Parameter(ParameterSetName = 'command')][string]$RemoteCommand,
  [Parameter(ParameterSetName = 'sftp')][switch]$Sftp,
  [Parameter(ParameterSetName = 'check')][switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$ssh = Join-Path $env:WINDIR 'System32\OpenSSH\ssh.exe'
$sftpExe = Join-Path $env:WINDIR 'System32\OpenSSH\sftp.exe'
$keyOnlyOptions = @(
  '-o', 'PasswordAuthentication=no',
  '-o', 'KbdInteractiveAuthentication=no',
  '-o', 'PreferredAuthentications=publickey',
  '-o', 'ConnectTimeout=15'
)

# The alias's IdentityFile is offered directly when an SSH agent is unavailable.
# Only public-key authentication is allowed, so no server password is requested.
if ($CheckOnly) {
  & $ssh @keyOnlyOptions $SshHost 'hostname; whoami'
  if ($LASTEXITCODE -ne 0) {
    throw "Key-only H200 login failed for '$SshHost'. Verify the alias and public-key registration."
  }
  return
}
if ($Sftp) {
  & $sftpExe @keyOnlyOptions $SshHost
  exit $LASTEXITCODE
}
if ($PSCmdlet.ParameterSetName -eq 'command') {
  & $ssh @keyOnlyOptions $SshHost $RemoteCommand
  exit $LASTEXITCODE
}

& $ssh @keyOnlyOptions -tt $SshHost
exit $LASTEXITCODE