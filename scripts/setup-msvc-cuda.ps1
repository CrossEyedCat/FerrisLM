# Activates Visual Studio 2026 (Community preferred) + CUDA 13.2 for candle-kernels build.
# Usage: . .\scripts\setup-msvc-cuda.ps1

$Vcvars = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'
$ClDir  = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\HostX64\x64'
$Cuda   = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'

if (-not (Test-Path $Vcvars)) {
    $Vcvars = 'C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat'
    $ClDir  = 'C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Tools\MSVC\14.51.36231\bin\HostX64\x64'
}
if (-not (Test-Path $Vcvars)) {
    throw "vcvars64.bat not found"
}

cmd.exe /c "`"$Vcvars`" && set" | ForEach-Object {
    if ($_ -match '^(?<key>[^=]+)=(?<val>.*)$') {
        Set-Item -Path "env:$($matches.key)" -Value $matches.val
    }
}

$env:CUDA_PATH = $Cuda
$env:PATH = "$Cuda\bin;$ClDir;$env:PATH"
Remove-Item Env:NVCC_CCBIN -ErrorAction SilentlyContinue
# Required for CUDA 13.2 + MSVC 2026 (CCCL)
$env:NVCC_PREPEND_FLAGS = '-Xcompiler "/Zc:preprocessor"'

Write-Host "MSVC: $(Get-Command cl.exe -ErrorAction Stop | Select-Object -ExpandProperty Source)"
Write-Host "NVCC: $(Get-Command nvcc.exe -ErrorAction Stop | Select-Object -ExpandProperty Source)"
