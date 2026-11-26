[CmdletBinding()]
param(
    [ValidateSet('CGBJ', 'RTX', 'Both')]
    [string]$Runtime = 'Both',

    [string]$BcpackTables,

    [switch]$Offline
)

$ErrorActionPreference = 'Stop'
$expectedHash = '1F1D4760419202BB5672F0C428CE51E456B5D08FBAEC8F371D043FD94E01C36B'
$tableDirectory = Join-Path $PSScriptRoot 'local'
$cachedTable = Join-Path $tableDirectory 'bcpack-tables.bin'
$tablePath = $cachedTable
if ($BcpackTables) {
    $tablePath = (Resolve-Path -LiteralPath $BcpackTables).Path
}

if (-not (Test-Path -LiteralPath $tablePath -PathType Leaf)) {
    if ($Offline) {
        throw 'Supply the known BCPack table file with -BcpackTables when building offline.'
    }
    New-Item -ItemType Directory -Force -Path $tableDirectory | Out-Null
    $download = Join-Path $tableDirectory ("bcpack-tables-{0}.part" -f [guid]::NewGuid())
    try {
        $url = 'https://raw.githubusercontent.com/Marvisak/VIctory/940d32f69b22265690a0b5a15d6ded572ba33baf/crates/victory-formats/src/bcpack-tables.bin'
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $download
        if ((Get-FileHash -LiteralPath $download).Hash -ne $expectedHash) {
            throw 'The downloaded BCPack table did not match the expected checksum.'
        }
        Move-Item -LiteralPath $download -Destination $cachedTable
    } finally {
        if (Test-Path -LiteralPath $download) {
            Remove-Item -LiteralPath $download
        }
    }
}

if ((Get-FileHash -LiteralPath $tablePath).Hash -ne $expectedHash) {
    throw 'The local BCPack table did not match the expected checksum.'
}

$cargo = (Get-Command cargo -CommandType Application).Source
$builds = @()
if ($Runtime -ne 'RTX') {
    $builds += @{ Directory = 'runtime/cgbj'; Arguments = @('--release') }
}
if ($Runtime -ne 'CGBJ') {
    $builds += @{ Directory = 'runtime/victory-rtx'; Arguments = @('--profile', 'diag', '--features', 'diagnostics') }
}

foreach ($build in $builds) {
    $directory = Join-Path $PSScriptRoot $build.Directory
    $localTable = Join-Path $directory 'crates/victory-formats/src/bcpack-tables.bin'
    if ($localTable -ne $tablePath) {
        Copy-Item -LiteralPath $tablePath -Destination $localTable
    }
    $arguments = @('build', '--locked', '-p', 'victory') + $build.Arguments
    if ($Offline) {
        $arguments += '--offline'
    }
    Push-Location -LiteralPath $directory
    try {
        & $cargo @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Build failed in $($build.Directory) with exit code $LASTEXITCODE."
        }
    } finally {
        Pop-Location
    }
}
