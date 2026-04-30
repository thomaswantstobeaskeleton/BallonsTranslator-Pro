param(
    [string]$TranslateDir = "C:\Users\thomas\BallonsTranslator-Pro\translate",
    [switch]$Recurse,
    [switch]$InstallPySide6IfMissing
)

$ErrorActionPreference = "Stop"

function Find-LReleaseCompiler {
    $commandNames = @(
        "pyside6-lrelease.exe",
        "pyside6-lrelease",
        "lrelease.exe",
        "lrelease",
        "lrelease-qt6.exe",
        "lrelease-qt5.exe",
        "pyside2-lrelease.exe",
        "pyside2-lrelease"
    )

    foreach ($name in $commandNames) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd -and $cmd.Source) {
            return $cmd.Source
        }
    }

    $roots = @(
        "$env:USERPROFILE\BallonsTranslator-Pro",
        "$env:USERPROFILE\BalloonsTranslator-Pro",
        "$env:LOCALAPPDATA\Programs\Python",
        "$env:APPDATA\Python",
        "C:\Qt",
        "C:\Program Files\Qt",
        "C:\Program Files (x86)\Qt"
    ) | Where-Object { $_ -and (Test-Path $_) }

    $fileNames = @(
        "pyside6-lrelease.exe",
        "lrelease.exe",
        "lrelease-qt6.exe",
        "lrelease-qt5.exe",
        "pyside2-lrelease.exe"
    )

    foreach ($root in $roots) {
        foreach ($fileName in $fileNames) {
            $hit = Get-ChildItem -Path $root -Filter $fileName -File -Recurse -ErrorAction SilentlyContinue |
                Sort-Object FullName -Descending |
                Select-Object -First 1

            if ($hit) {
                return $hit.FullName
            }
        }
    }

    return $null
}

function Try-InstallPySide6 {
    Write-Host "Qt lrelease was not found. Trying to install PySide6, which includes pyside6-lrelease..." -ForegroundColor Yellow

    $pythonLaunchers = @("py", "python", "python3")
    foreach ($launcher in $pythonLaunchers) {
        $cmd = Get-Command $launcher -ErrorAction SilentlyContinue
        if ($cmd) {
            & $launcher -m pip install --upgrade PySide6
            if ($LASTEXITCODE -eq 0) {
                return $true
            }
        }
    }

    return $false
}

if (-not (Test-Path $TranslateDir)) {
    Write-Error "Translation folder not found: $TranslateDir"
    exit 1
}

$compiler = Find-LReleaseCompiler

if (-not $compiler -and $InstallPySide6IfMissing) {
    $installed = Try-InstallPySide6
    if ($installed) {
        $compiler = Find-LReleaseCompiler
    }
}

if (-not $compiler) {
    Write-Host ""
    Write-Host "No .ts -> .qm compiler found." -ForegroundColor Red
    Write-Host "Install one of these, then run this script again:"
    Write-Host "  1. PySide6:  py -m pip install PySide6"
    Write-Host "  2. Qt: install Qt, making sure lrelease.exe is included"
    Write-Host ""
    Write-Host "Or run this script with:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\Compile-QtTranslations.ps1 -InstallPySide6IfMissing"
    exit 2
}

Write-Host "Using compiler: $compiler" -ForegroundColor Green
Write-Host "Translation folder: $TranslateDir"

if ($Recurse) {
    $tsFiles = Get-ChildItem -Path $TranslateDir -Filter "*.ts" -File -Recurse
} else {
    $tsFiles = Get-ChildItem -Path $TranslateDir -Filter "*.ts" -File
}

if (-not $tsFiles -or $tsFiles.Count -eq 0) {
    Write-Host "No .ts files found." -ForegroundColor Yellow
    exit 0
}

$success = 0
$failed = 0

foreach ($ts in $tsFiles) {
    $qm = [System.IO.Path]::ChangeExtension($ts.FullName, ".qm")

    Write-Host ""
    Write-Host "Compiling: $($ts.FullName)"
    & $compiler $ts.FullName -qm $qm

    if ($LASTEXITCODE -eq 0 -and (Test-Path $qm)) {
        Write-Host "Created:   $qm" -ForegroundColor Green
        $success += 1
    } else {
        Write-Host "FAILED:    $($ts.FullName)" -ForegroundColor Red
        $failed += 1
    }
}

Write-Host ""
Write-Host "Done. Succeeded: $success  Failed: $failed"

if ($failed -gt 0) {
    exit 3
}
