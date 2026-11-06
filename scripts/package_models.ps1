$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$files = @(
    'models/mask2former-swin-large-ade-semantic/config.json',
    'models/mask2former-swin-large-ade-semantic/preprocessor_config.json',
    'models/mask2former-swin-large-ade-semantic/model.safetensors',
    'models/mask2former_hazard_head.pt',
    'models/hazard_best.pt'
)
foreach ($relative in $files) {
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot $relative) -PathType Leaf)) {
        throw "Missing model file: $relative"
    }
}
$outputDirectory = Join-Path $projectRoot 'outputs/sharing'
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$zipPath = Join-Path $outputDirectory ('stair-models-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.zip')
$codeCommit = & git -c "safe.directory=$projectRoot" -C $projectRoot rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Cannot read code commit' }
$metadata = [ordered]@{
    created_at = (Get-Date).ToString('o')
    code_commit = $codeCommit
    extraction = 'Place models next to app.py in the cloned stair repository.'
    files = @($files | ForEach-Object {
        $absolute = Join-Path $projectRoot $_
        [ordered]@{path=$_;bytes=(Get-Item -LiteralPath $absolute).Length;sha256=(Get-FileHash -LiteralPath $absolute -Algorithm SHA256).Hash.ToLowerInvariant()}
    })
}
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$stream = [IO.File]::Open($zipPath,[IO.FileMode]::CreateNew)
$archive = [IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($relative in $files) {
        [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive,(Join-Path $projectRoot $relative),$relative,[IO.Compression.CompressionLevel]::Fastest) | Out-Null
    }
    $entry = $archive.CreateEntry('MODEL_FILES.json')
    $writer = [IO.StreamWriter]::new($entry.Open(),[Text.UTF8Encoding]::new($false))
    try { $writer.Write(($metadata | ConvertTo-Json -Depth 6)) } finally { $writer.Dispose() }
} finally {
    $archive.Dispose()
    $stream.Dispose()
}
$hash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText(($zipPath+'.sha256.txt'),($hash+'  '+[IO.Path]::GetFileName($zipPath)+[Environment]::NewLine),[Text.UTF8Encoding]::new($false))
Write-Output "Package: $zipPath"
Write-Output "Bytes: $((Get-Item -LiteralPath $zipPath).Length)"
Write-Output "SHA256: $hash"
