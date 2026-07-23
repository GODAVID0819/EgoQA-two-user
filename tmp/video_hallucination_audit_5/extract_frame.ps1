param(
    [Parameter(Mandatory = $true)]
    [string]$InputVideo,
    [Parameter(Mandatory = $true)]
    [string]$OutputImage,
    [Parameter(Mandatory = $true)]
    [double]$CenterSeconds
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Runtime.WindowsRuntime

[Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.StorageFolder, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.CreationCollisionOption, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
[Windows.Media.Transcoding.MediaTranscoder, Windows.Media.Transcoding, ContentType = WindowsRuntime] | Out-Null
[Windows.Media.Transcoding.PrepareTranscodeResult, Windows.Media.Transcoding, ContentType = WindowsRuntime] | Out-Null
[Windows.Media.MediaProperties.MediaEncodingProfile, Windows.Media.MediaProperties, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.FileProperties.StorageItemThumbnail, Windows.Storage.FileProperties, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.FileProperties.ThumbnailMode, Windows.Storage.FileProperties, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.FileProperties.ThumbnailOptions, Windows.Storage.FileProperties, ContentType = WindowsRuntime] | Out-Null

$asyncResultMethod = [System.WindowsRuntimeSystemExtensions].GetMethods() |
    Where-Object {
        $_.Name -eq "AsTask" -and
        $_.IsGenericMethod -and
        $_.GetGenericArguments().Count -eq 1 -and
        $_.GetParameters().Count -eq 1 -and
        $_.ReturnType.IsGenericType
    } |
    Select-Object -First 1

$asyncProgressMethod = [System.WindowsRuntimeSystemExtensions].GetMethods() |
    Where-Object {
        $_.Name -eq "AsTask" -and
        $_.IsGenericMethod -and
        $_.GetGenericArguments().Count -eq 1 -and
        $_.GetParameters().Count -eq 1 -and
        -not $_.ReturnType.IsGenericType
    } |
    Select-Object -First 1

function Await-Result($Operation, [Type]$ResultType) {
    $task = $asyncResultMethod.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $task.Wait()
    return $task.Result
}

function Await-Progress($Operation, [Type]$ProgressType) {
    $task = $asyncProgressMethod.MakeGenericMethod($ProgressType).Invoke($null, @($Operation))
    $task.Wait()
}

$resolvedInput = (Resolve-Path -LiteralPath $InputVideo).Path
$outputDirectory = Split-Path -Parent $OutputImage
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$resolvedOutputDirectory = (Resolve-Path -LiteralPath $outputDirectory).Path
$outputLeaf = Split-Path -Leaf $OutputImage
$segmentLeaf = [System.IO.Path]::GetFileNameWithoutExtension($outputLeaf) + ".segment.mp4"
$segmentPath = Join-Path $resolvedOutputDirectory $segmentLeaf

$inputFile = Await-Result (
    [Windows.Storage.StorageFile]::GetFileFromPathAsync($resolvedInput)
) ([Windows.Storage.StorageFile])
$outputFolder = Await-Result (
    [Windows.Storage.StorageFolder]::GetFolderFromPathAsync($resolvedOutputDirectory)
) ([Windows.Storage.StorageFolder])
$segmentFile = Await-Result (
    $outputFolder.CreateFileAsync(
        $segmentLeaf,
        [Windows.Storage.CreationCollisionOption]::ReplaceExisting
    )
) ([Windows.Storage.StorageFile])
$profile = Await-Result (
    [Windows.Media.MediaProperties.MediaEncodingProfile]::CreateFromFileAsync($inputFile)
) ([Windows.Media.MediaProperties.MediaEncodingProfile])

$startSeconds = [Math]::Max(0.0, $CenterSeconds - 1.0)
$stopSeconds = [Math]::Min(30.0, $CenterSeconds + 1.0)
$transcoder = [Windows.Media.Transcoding.MediaTranscoder]::new()
$transcoder.AlwaysReencode = $true
$transcoder.HardwareAccelerationEnabled = $false
$transcoder.TrimStartTime = [TimeSpan]::FromSeconds($startSeconds)
$transcoder.TrimStopTime = [TimeSpan]::FromSeconds($stopSeconds)

$prepared = Await-Result (
    $transcoder.PrepareFileTranscodeAsync($inputFile, $segmentFile, $profile)
) ([Windows.Media.Transcoding.PrepareTranscodeResult])
if (-not $prepared.CanTranscode) {
    throw "Cannot transcode $resolvedInput at $CenterSeconds seconds: $($prepared.FailureReason)"
}
Await-Progress ($prepared.TranscodeAsync()) ([double])

$thumbnail = Await-Result (
    $segmentFile.GetThumbnailAsync(
        [Windows.Storage.FileProperties.ThumbnailMode]::VideosView,
        [uint32]1200,
        [Windows.Storage.FileProperties.ThumbnailOptions]::ResizeThumbnail
    )
) ([Windows.Storage.FileProperties.StorageItemThumbnail])

$sourceStream = [System.IO.WindowsRuntimeStreamExtensions]::AsStreamForRead($thumbnail)
$destinationStream = [System.IO.File]::Create((Join-Path $resolvedOutputDirectory $outputLeaf))
try {
    $sourceStream.CopyTo($destinationStream)
}
finally {
    $destinationStream.Dispose()
    $sourceStream.Dispose()
    $thumbnail.Dispose()
}

Write-Output "$OutputImage|center_seconds=$CenterSeconds|segment=$segmentPath"
