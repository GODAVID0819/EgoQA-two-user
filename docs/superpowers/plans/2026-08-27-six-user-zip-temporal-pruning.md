# Six-User ZIP Temporal Pruning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `multi-user.zip` 的时间感知 K-means、center-gap hard gate 和双侧 pair 剪枝接入六用户候选路径，同时让最终 QA 始终使用完整 speaker 原视频。

**Architecture:** 原样加入 ZIP 的两个 sidecar 模块，生产代码通过函数内延迟导入调用 `time_aware_clustered_frame_representatives` 与 `prune_time_aware_cluster_pair`，避免 sidecar 反向导入 `group_relative_clip_sampling` 造成模块初始化循环。每个 speaker 分别形成五个 ZIP pair 结果；provider 使用右侧剪枝区间，speaker 只保留左侧诊断并在媒体物化时使用完整原视频。

**Tech Stack:** Python 3.12、NumPy、`unittest`、现有 CLIP embedding 与 FFmpeg 物化辅助函数。

---

## 文件结构

- 新增 `temporal_kmeans_grid_sidecar.py`：从 ZIP 原样恢复时间感知聚类、gap、pair 剪枝与离线 grid。
- 新增 `cross_user_temporal_gate_grid_sidecar.py`：从 ZIP 原样恢复 center/interval gate grid。
- 修改 `group_relative_clip_sampling.py`：增加六用户 ZIP pair 聚合器，切换六用户正式调用，保留旧函数作为历史实现。
- 新增 `tests/test_zip_temporal_pruning.py`：锁定 ZIP 算法核心和生产固定参数。
- 修改 `tests/test_six_user_group_relative_sampling.py`：锁定五 pair 聚合与 speaker 完整媒体路由。
- 修改 `tests/test_three_minute_blockwise_pruning.py`：把旧 30 秒 blockwise 生产预期改成 ZIP 全窗口 `K=72` 预期；保留视频拼接缓存测试。
- 不修改 `video_qa_loop.py`、prompt、GRPO、DPO、reviewer、optimizer、checkpoint 或 `.sbatch`。

### Task 1: 创建可恢复备份

**Files:**
- Copy: `group_relative_clip_sampling.py`
- Copy: `tests/test_six_user_group_relative_sampling.py`
- Copy: `tests/test_three_minute_blockwise_pruning.py`
- Create outside repo: `../collaborator_delivery/backups/six_user_zip_temporal_pruning_<timestamp>/manifest.csv`

- [ ] **Step 1: 解析并验证备份目录**

运行 PowerShell，确认目标位于工作区的 `collaborator_delivery/backups` 内：

```powershell
$repo = (Resolve-Path '.').Path
$workspace = (Resolve-Path '..').Path
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$backup = Join-Path $workspace "collaborator_delivery/backups/six_user_zip_temporal_pruning_$stamp"
$allowed = Join-Path $workspace 'collaborator_delivery/backups'
$backupFull = [IO.Path]::GetFullPath($backup)
$allowedFull = [IO.Path]::GetFullPath($allowed)
if (-not $backupFull.StartsWith($allowedFull, [StringComparison]::OrdinalIgnoreCase)) {
    throw "backup target escaped allowed root: $backupFull"
}
```

- [ ] **Step 2: 复制当前 dirty 文件并生成 SHA-256 manifest**

```powershell
$sources = @(
    'group_relative_clip_sampling.py',
    'tests/test_six_user_group_relative_sampling.py',
    'tests/test_three_minute_blockwise_pruning.py'
)
New-Item -ItemType Directory -Force -Path $backupFull | Out-Null
$rows = foreach ($relative in $sources) {
    $source = Join-Path $repo $relative
    if (-not (Test-Path -LiteralPath $source)) { continue }
    $target = Join-Path $backupFull $relative
    New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
    Copy-Item -LiteralPath $source -Destination $target
    $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLowerInvariant()
    $targetHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToLowerInvariant()
    if ($sourceHash -ne $targetHash) { throw "backup hash mismatch: $relative" }
    [pscustomobject]@{ relative_path=$relative; sha256=$sourceHash; bytes=(Get-Item $source).Length }
}
$rows | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath (Join-Path $backupFull 'manifest.csv')
Write-Output "BACKUP_READY=$backupFull"
```

预期：输出唯一 `BACKUP_READY=...`，manifest 中每个已存在源文件都有一行且源/备份 hash 相同。

### Task 2: 先锁定 ZIP 算法核心

**Files:**
- Create: `tests/test_zip_temporal_pruning.py`
- Create after RED: `temporal_kmeans_grid_sidecar.py`
- Create after RED: `cross_user_temporal_gate_grid_sidecar.py`

- [ ] **Step 1: 写入失败测试**

测试使用现有测试的 package shim，并锁定 `w=0.1`、center `G=10`、阈值边界和全窗口 K：

```python
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if "egolife_two_user_qa" not in sys.modules:
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(ROOT)]
    sys.modules["egolife_two_user_qa"] = package

from egolife_two_user_qa import cross_user_temporal_gate_grid_sidecar as gate_grid
from egolife_two_user_qa import temporal_kmeans_grid_sidecar as sidecar


class ZipTemporalPruningTests(unittest.TestCase):
    def test_zero_weight_matches_current_cosine_clustering(self) -> None:
        embeddings = [[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.02, 0.98]]
        sidecar.assert_zero_weight_compatibility(
            embeddings, 2, timestamps_seconds=[0.0, 1.0, 20.0, 21.0]
        )

    def test_center_gate_uses_inclusive_ten_second_boundary(self) -> None:
        variants = gate_grid.build_cross_user_gap_variants(
            [180.0], [2.5], [0.82], ["center"], [10.0], within_time_weight=0.1
        )
        selected = [row for row in variants if row["variant_kind"] == "temporal_center_gate"]
        self.assertEqual(selected[0]["k"], 72)
        self.assertEqual(selected[0]["max_cross_gap_seconds"], 10.0)
        self.assertEqual(selected[0]["within_time_weight"], 0.1)

    def test_center_gap_is_distinct_from_interval_gap(self) -> None:
        left = {"timestamp_seconds": 2.0, "member_timestamps": [0.0, 4.0], "temporal_center_seconds": 2.0}
        right = {"timestamp_seconds": 10.0, "member_timestamps": [4.0, 16.0], "temporal_center_seconds": 10.0}
        gaps = sidecar.cross_cluster_temporal_gaps(left, right)
        self.assertEqual(gaps["center_gap_seconds"], 8.0)
        self.assertEqual(gaps["interval_gap_seconds"], 0.0)
```

- [ ] **Step 2: 运行测试并验证 RED**

Run: `python -m unittest tests.test_zip_temporal_pruning -v`

Expected: FAIL，原因是 `temporal_kmeans_grid_sidecar` 或 `cross_user_temporal_gate_grid_sidecar` 尚不存在，而不是测试语法错误。

- [ ] **Step 3: 从 ZIP 逐字节提取两个模块**

```powershell
$zipPath = (Resolve-Path '..\multi-user.zip').Path
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [IO.Compression.ZipFile]::OpenRead($zipPath)
try {
    foreach ($mapping in @(
        @('multi-user/temporal_kmeans_grid_sidecar.py', 'temporal_kmeans_grid_sidecar.py'),
        @('multi-user/cross_user_temporal_gate_grid_sidecar.py', 'cross_user_temporal_gate_grid_sidecar.py')
    )) {
        $entry = $archive.GetEntry($mapping[0])
        if ($null -eq $entry) { throw "ZIP entry missing: $($mapping[0])" }
        $input = $entry.Open()
        $output = [IO.File]::Create((Join-Path (Resolve-Path '.').Path $mapping[1]))
        try { $input.CopyTo($output) } finally { $output.Dispose(); $input.Dispose() }
    }
} finally {
    $archive.Dispose()
}
```

验证 SHA-256：

```powershell
(Get-FileHash -Algorithm SHA256 temporal_kmeans_grid_sidecar.py).Hash.ToLowerInvariant()
(Get-FileHash -Algorithm SHA256 cross_user_temporal_gate_grid_sidecar.py).Hash.ToLowerInvariant()
```

Expected:

- `611ad20d86a8b26fcaa892ba9b26c14ddba5e8aaf80ff5daad35ef086e9e9e56`
- `6c0463702576f4a41336101bb26168541744255b80c75b75a8e5c8938f8e11c2`

- [ ] **Step 4: 运行核心测试并验证 GREEN**

Run: `python -m unittest tests.test_zip_temporal_pruning -v`

Expected: 3 tests PASS。

### Task 3: 先锁定六用户五 pair 聚合合同

**Files:**
- Modify: `tests/test_six_user_group_relative_sampling.py`
- Modify after RED: `group_relative_clip_sampling.py`

- [ ] **Step 1: 添加失败测试**

测试通过 patch ZIP 函数隔离媒体与 NumPy 细节，要求每个 speaker 调用五个 pair、provider 使用右侧区间、speaker 汇总保持完整媒体：

```python
def test_zip_temporal_pruning_builds_five_pairs_and_keeps_full_speaker(self) -> None:
    frames = [self.frames(3) for _ in range(6)]
    embeddings = [[[1.0, 0.0]] * 3 for _ in range(6)]

    clusters = self.cluster_result([[1.0, 0.0]], [[0, 1, 2]])
    clusters["clustering"] = {"time_weight": 0.1, "temporal_unit_seconds": 30.0}

    def pair_result(*args, **kwargs):
        return {
            "passed": True,
            "high_similarity_representative_pairs": [
                {"left_cluster_index": 0, "right_cluster_index": 0, "similarity": 0.9}
            ],
            "left_marked_frame_indices": [0],
            "right_marked_frame_indices": [1],
            "left_remove_intervals": [[0.5, 1.5]],
            "right_remove_intervals": [[1.5, 2.5]],
            "left_keep_intervals": [[0.0, 0.5], [1.5, 30.0]],
            "right_keep_intervals": [[0.0, 1.5], [2.5, 30.0]],
            "left_kept_duration_seconds": 29.0,
            "right_kept_duration_seconds": 29.0,
            "left_removed_duration_seconds": 1.0,
            "right_removed_duration_seconds": 1.0,
            "high_similarity_representative_pair_count": 1,
        }

    with mock.patch(
        "egolife_two_user_qa.temporal_kmeans_grid_sidecar.time_aware_clustered_frame_representatives",
        return_value=clusters,
    ) as cluster, mock.patch(
        "egolife_two_user_qa.temporal_kmeans_grid_sidecar.prune_time_aware_cluster_pair",
        side_effect=pair_result,
    ) as prune:
        result = group_relative_clip_sampling.clustered_six_user_zip_temporal_pruning(
            frames,
            embeddings,
            speaker_index=0,
            start_seconds=0.0,
            duration_seconds=30.0,
            sample_interval_seconds=1.0,
        )

    self.assertEqual(cluster.call_count, 6)
    self.assertEqual(prune.call_count, 5)
    self.assertEqual(len(result["pair_results"]), 5)
    self.assertEqual(result["videos"][0]["keep_intervals"], [[0.0, 30.0]])
    self.assertEqual(result["videos"][1]["remove_intervals"], [[1.5, 2.5]])
    self.assertTrue(result["passed"])
```

- [ ] **Step 2: 运行单测并验证 RED**

Run: `python -m unittest tests.test_six_user_group_relative_sampling.SpeakerProviderAllPairsPruningTests.test_zip_temporal_pruning_builds_five_pairs_and_keeps_full_speaker -v`

Expected: FAIL with `AttributeError: ... clustered_six_user_zip_temporal_pruning`。

- [ ] **Step 3: 实现最小六用户适配器**

在 `group_relative_clip_sampling.py` 新增以下接口。函数内延迟导入是必需的，因为 ZIP sidecar 顶层会反向导入本模块中的保护与矩阵辅助函数。

```python
def clustered_six_user_zip_temporal_pruning(
    frames_by_video: list[list[dict[str, Any]]],
    embeddings_by_video: list[list[list[float]]],
    *,
    speaker_index: int,
    start_seconds: float,
    duration_seconds: float,
    sample_interval_seconds: float,
    seconds_per_cluster: float = 2.5,
    time_weight: float = 0.1,
    temporal_unit_seconds: float = 30.0,
    max_iterations: int = 25,
    high_similarity_threshold: float = 0.82,
    cross_gap_mode: str = "center",
    max_cross_gap_seconds: float = 10.0,
    min_pruned_video_seconds: float = 8.0,
    min_pruned_video_percent: float = 20.0,
) -> dict[str, Any]:
    from .temporal_kmeans_grid_sidecar import (
        prune_time_aware_cluster_pair,
        time_aware_clustered_frame_representatives,
    )

    if len(frames_by_video) != 6 or len(embeddings_by_video) != 6:
        raise ValueError("ZIP temporal pruning requires exactly 6 videos")
    if speaker_index not in range(6):
        raise ValueError("speaker_index must be between 0 and 5")
    if duration_seconds <= 0 or sample_interval_seconds <= 0 or seconds_per_cluster <= 0:
        raise ValueError("duration, sample interval, and seconds per cluster must be positive")

    cluster_count = max(1, math.ceil(duration_seconds / seconds_per_cluster))
    clusters_by_video = [
        time_aware_clustered_frame_representatives(
            frames,
            embeddings,
            cluster_count=cluster_count,
            time_weight=time_weight,
            temporal_unit_seconds=temporal_unit_seconds,
            max_iterations=max_iterations,
        )
        for frames, embeddings in zip(frames_by_video, embeddings_by_video)
    ]
    provider_indices = [index for index in range(6) if index != speaker_index]
    pair_results = []
    videos = [None] * 6
    videos[speaker_index] = {
        "video_index": speaker_index,
        "clusters": clusters_by_video[speaker_index]["representatives"],
        "marked_cluster_indices": [],
        "marked_frame_indices": [],
        "trigger_event_indices": [],
        "keep_intervals": [[start_seconds, start_seconds + duration_seconds]],
        "remove_intervals": [],
        "kept_duration_seconds": duration_seconds,
        "removed_duration_seconds": 0.0,
        "passed": True,
    }

    for provider_index in provider_indices:
        full_matrix = frame_similarity_matrix(
            embeddings_by_video[speaker_index], embeddings_by_video[provider_index]
        )
        pruning = prune_time_aware_cluster_pair(
            frames_by_video[speaker_index],
            frames_by_video[provider_index],
            embeddings_by_video[speaker_index],
            embeddings_by_video[provider_index],
            clusters_by_video[speaker_index],
            clusters_by_video[provider_index],
            full_frame_matrix=full_matrix,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            sample_interval_seconds=sample_interval_seconds,
            high_similarity_threshold=high_similarity_threshold,
            min_pruned_video_seconds=min_pruned_video_seconds,
            pruning_protection_mode="min_percent",
            min_pruned_video_percent=min_pruned_video_percent,
            cross_gap_mode=cross_gap_mode,
            max_cross_gap_seconds=max_cross_gap_seconds,
        )
        pair_results.append(
            {"speaker_index": speaker_index, "provider_index": provider_index, "pruning": pruning}
        )
        videos[provider_index] = {
            "video_index": provider_index,
            "clusters": clusters_by_video[provider_index]["representatives"],
            "marked_cluster_indices": sorted(
                {
                    int(row["right_cluster_index"])
                    for row in pruning["high_similarity_representative_pairs"]
                }
            ),
            "marked_frame_indices": list(pruning["right_marked_frame_indices"]),
            "trigger_event_indices": [len(pair_results) - 1],
            "remove_intervals": list(pruning["right_remove_intervals"]),
            "keep_intervals": list(pruning["right_keep_intervals"]),
            "kept_duration_seconds": pruning["right_kept_duration_seconds"],
            "removed_duration_seconds": pruning["right_removed_duration_seconds"],
            "passed": bool(pruning["passed"]),
        }

    return {
        "method": "zip_temporal_kmeans_center_gate_pair_pruning_v1",
        "speaker_index": speaker_index,
        "provider_indices": provider_indices,
        "cluster_count": cluster_count,
        "time_weight": time_weight,
        "temporal_unit_seconds": temporal_unit_seconds,
        "cross_gap_mode": cross_gap_mode,
        "max_cross_gap_seconds": max_cross_gap_seconds,
        "pair_results": pair_results,
        "events": [
            pair
            for pair in pair_results
            if pair["pruning"]["high_similarity_representative_pair_count"] > 0
        ],
        "videos": videos,
        "passed": all(pair["pruning"]["passed"] for pair in pair_results),
}
```

- [ ] **Step 4: 运行测试并验证 GREEN**

Run: `python -m unittest tests.test_six_user_group_relative_sampling.SpeakerProviderAllPairsPruningTests.test_zip_temporal_pruning_builds_five_pairs_and_keeps_full_speaker -v`

Expected: PASS。

### Task 4: 切换六用户正式路径到全窗口 ZIP 剪枝

**Files:**
- Modify: `tests/test_three_minute_blockwise_pruning.py`
- Modify after RED: `group_relative_clip_sampling.py`

- [ ] **Step 1: 把旧 blockwise 生产测试改成全窗口失败测试**

```python
def test_three_minute_analysis_uses_zip_full_window_k72_for_every_speaker(tmp_path: Path) -> None:
    rows = [
        {
            "user": name,
            "clip": {
                "agent_name": name,
                "agent_dir": agent_dir,
                "local_video": str(tmp_path / f"{agent_dir}.mp4"),
            },
            "frames": [
                {"timestamp_seconds": float(second), "path": f"{agent_dir}-{second}.jpg"}
                for second in range(180)
            ],
        }
        for agent_dir, _agent_id, name in AGENTS
    ]

    class Encoder:
        model_id = "fake/clip"

        def encode(self, paths):
            return [[1.0, 0.0] for _ in paths]

    calls = []

    def zip_pruning(*_args, speaker_index, **kwargs):
        calls.append(kwargs)
        return {
            "method": "zip_temporal_kmeans_center_gate_pair_pruning_v1",
            "speaker_index": speaker_index,
            "passed": False,
            "events": [],
            "videos": [],
            "pair_results": [],
        }

    group = {
        "day": "DAY1",
        "time_token": "12000000",
        "clips": [row["clip"] for row in rows],
    }
    with (
        mock.patch.object(sampling, "group_clip_frames", return_value=rows),
        mock.patch.object(
            sampling,
            "clustered_six_user_zip_temporal_pruning",
            side_effect=zip_pruning,
        ) as zip_kernel,
        mock.patch.object(
            sampling,
            "blockwise_speaker_provider_all_pairs_pruning",
        ) as blockwise,
    ):
        sampling.analyze_group_relative_similarity(
            group,
            output_dir=tmp_path / "output",
            cache_dir=tmp_path / "cache",
            encoder=Encoder(),
            duration_seconds=180.0,
            pruning_block_seconds=30.0,
            selected_count=6,
        )

    assert zip_kernel.call_count == 6
    blockwise.assert_not_called()
    assert len(calls) == 6
    assert all(call["seconds_per_cluster"] == 2.5 for call in calls)
    assert all(call["time_weight"] == 0.1 for call in calls)
    assert all(call["cross_gap_mode"] == "center" for call in calls)
    assert all(call["max_cross_gap_seconds"] == 10.0 for call in calls)
```

- [ ] **Step 2: 运行测试并验证 RED**

Run: `python -m pytest tests/test_three_minute_blockwise_pruning.py::test_three_minute_analysis_uses_zip_full_window_k72_for_every_speaker -q`

Expected: FAIL，因为当前 `analyze_group_relative_similarity` 仍调用 blockwise 内核。

- [ ] **Step 3: 切换正式调用并传播固定参数**

给 `analyze_group_relative_similarity` 和 `mine_group_relative_clip_candidates` 增加以下默认参数并逐层原样传递：

```python
pruning_seconds_per_cluster: float = 2.5,
pruning_time_weight: float = 0.1,
pruning_temporal_unit_seconds: float = 30.0,
pruning_max_iterations: int = 25,
pruning_cross_gap_mode: str = "center",
pruning_max_cross_gap_seconds: float = 10.0,
min_pruned_video_percent: float | None = 20.0,
```

六用户分支无条件调用：

```python
consensus = clustered_six_user_zip_temporal_pruning(
    [row["frames"] for row in rows],
    frame_embeddings_by_clip,
    speaker_index=speaker_index,
    start_seconds=start_seconds,
    duration_seconds=duration_seconds,
    sample_interval_seconds=sample_interval_seconds,
    seconds_per_cluster=pruning_seconds_per_cluster,
    time_weight=pruning_time_weight,
    temporal_unit_seconds=pruning_temporal_unit_seconds,
    max_iterations=pruning_max_iterations,
    high_similarity_threshold=high_similarity_interval_threshold,
    cross_gap_mode=pruning_cross_gap_mode,
    max_cross_gap_seconds=pruning_max_cross_gap_seconds,
    min_pruned_video_seconds=min_pruned_video_seconds,
    min_pruned_video_percent=float(min_pruned_video_percent or 20.0),
)
```

保留 `pruning_block_seconds` 和 `pruning_clusters_per_video` 参数以兼容现有调用者，但六用户 ZIP 路径不再读取它们；两用户路径保持原行为。

- [ ] **Step 4: 更新内置 CLI 参数**

新增并传递：

```python
parser.add_argument("--pruning-seconds-per-cluster", type=float, default=2.5)
parser.add_argument("--pruning-time-weight", type=float, default=0.1)
parser.add_argument("--pruning-temporal-unit-seconds", type=float, default=30.0)
parser.add_argument("--pruning-max-iterations", type=int, default=25)
parser.add_argument("--pruning-cross-gap-mode", choices=["center", "interval"], default="center")
parser.add_argument("--pruning-max-cross-gap-seconds", type=float, default=10.0)
```

现有 `--pruning-block-seconds` 和 `--pruning-clusters-per-video` 的 help 标明：仅旧路径兼容，六用户 ZIP 路径使用 seconds-per-cluster。

- [ ] **Step 5: 运行三分钟目标测试并验证 GREEN**

Run: `python -m pytest tests/test_three_minute_blockwise_pruning.py -q`

Expected: 所有保留的视频拼接测试和新的全窗口 ZIP 测试 PASS；旧 blockwise 生产断言已被替换，不把历史行为当作当前合同。

### Task 5: 锁定完整 speaker QA 媒体路由

**Files:**
- Modify: `tests/test_six_user_group_relative_sampling.py`
- Modify only if test exposes a regression: `group_relative_clip_sampling.py`

- [ ] **Step 1: 添加媒体路由测试**

```python
def test_materializer_ignores_speaker_pair_pruning_and_uses_original_video(self) -> None:
    rows = self.six_rows()
    consensus = {
        "method": "zip_temporal_kmeans_center_gate_pair_pruning_v1",
        "speaker_index": 0,
        "pair_results": [
            {
                "speaker_index": 0,
                "provider_index": 1,
                "pruning": {"left_remove_intervals": [[0.5, 1.5]]},
            }
        ],
        "videos": [
            {
                "video_index": index,
                "keep_intervals": [[0.0, 30.0]] if index == 0 else [[0.0, 1.5], [2.5, 30.0]],
                "remove_intervals": [] if index == 0 else [[1.5, 2.5]],
                "marked_cluster_indices": [],
                "trigger_event_indices": [],
                "kept_duration_seconds": 30.0 if index == 0 else 29.0,
                "passed": True,
            }
            for index in range(6)
        ],
        "passed": True,
    }

    def fake_materialize(clip, *, media_role, keep_intervals, **kwargs):
        return {
            **clip,
            "media_role": media_role,
            "is_pruned": keep_intervals is not None,
            "received_keep_intervals": keep_intervals,
        }

    with mock.patch.object(
        group_relative_clip_sampling,
        "_materialize_six_user_clip",
        side_effect=fake_materialize,
    ):
        clips = group_relative_clip_sampling.materialize_six_user_consensus_candidate(
            rows,
            consensus,
            output_dir=self.tmp_path / "zip_temporal",
            ffmpeg_binary="ffmpeg",
        )

    self.assertEqual(clips[0]["local_video"], rows[0]["clip"]["local_video"])
    self.assertEqual(clips[0]["media_role"], "speaker_reference_unpruned")
    self.assertFalse(clips[0]["is_pruned"])
    self.assertTrue(all(clip["is_pruned"] for clip in clips[1:]))
```

- [ ] **Step 2: 运行测试并观察结果**

Run: `python -m unittest tests.test_six_user_group_relative_sampling.SixUserRoleSelectionTests.test_materializer_ignores_speaker_pair_pruning_and_uses_original_video -v`

Expected: 若当前物化器已保持 speaker 原视频则直接 PASS；若失败，只修复 speaker 的 `keep_intervals=None` 路由，不修改 ZIP pair 诊断。

- [ ] **Step 3: 运行六用户媒体回归**

Run: `python -m unittest tests.test_six_user_group_relative_sampling -v`

Expected: 新 ZIP 合同测试 PASS；与已被取代的 provider-only/blockwise 断言冲突的测试已明确改写，其余两用户与媒体排序测试继续 PASS。

### Task 6: 完整本地验证与差异审计

**Files:**
- Verify only; no new production files unless a targeted test reveals a defect.

- [ ] **Step 1: 编译检查**

Run:

```powershell
python -m py_compile temporal_kmeans_grid_sidecar.py cross_user_temporal_gate_grid_sidecar.py group_relative_clip_sampling.py tests/test_zip_temporal_pruning.py
```

Expected: exit code 0，无 traceback。

- [ ] **Step 2: 运行核心 ZIP 与六用户测试**

Run:

```powershell
python -m unittest tests.test_zip_temporal_pruning tests.test_six_user_group_relative_sampling -v
python -m pytest tests/test_three_minute_blockwise_pruning.py -q
```

Expected: 0 failures、0 errors。

- [ ] **Step 3: 运行相邻 QA 回归**

Run:

```powershell
python -m unittest tests.test_six_user_prompts tests.test_six_user_video_qa_loop -v
```

Expected: 0 failures、0 errors；这只证明本地 prompt/media 路由未发生已知回归。

- [ ] **Step 4: 检查 ZIP 文件来源与工作树差异**

Run:

```powershell
Get-FileHash -Algorithm SHA256 temporal_kmeans_grid_sidecar.py,cross_user_temporal_gate_grid_sidecar.py
git diff --check
git status --short
git diff -- group_relative_clip_sampling.py tests/test_six_user_group_relative_sampling.py tests/test_three_minute_blockwise_pruning.py
```

Expected: 两个 sidecar hash 与 Task 2 相同；`git diff --check` 无错误；不出现 GRPO、DPO、reviewer、optimizer、checkpoint 或 `.sbatch` 新修改。

- [ ] **Step 5: 不提交重叠 dirty 实现文件**

`group_relative_clip_sampling.py` 和相关测试在本任务开始前已有未提交修改。完成后只报告 diff、备份路径和验证证据，不自动提交这些重叠文件，避免把用户既有工作错误归入本次提交。

## 证据边界

完成以上步骤后只能声称：本地代码与单元测试实现了 ZIP 算法、固定 `w=0.1`、center `G=10`、全窗口 K、五 pair 双侧判定和完整 speaker QA 路由。没有 Torch 运行、远端产物、grid 比较或人工 QA 复核时，仍必须明确报告“远端未验证，`G=10` 未证明最优”。
