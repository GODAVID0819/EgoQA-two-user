"""Project EgoLife/Aria CPF gaze rays onto RGB image coordinates.

EgoLife EyeGaze CSVs contain Project Aria MPS gaze in the Central Pupil
Frame (CPF), not image-space pixels. This module only emits 2D gaze points
when an explicit camera calibration is supplied.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


Matrix4 = list[list[float]]


@dataclass(frozen=True)
class AriaProjectionCalibration:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int | None
    height: int | None
    t_camera_cpf: Matrix4
    source_path: str


def _as_float(value: Any, *, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Calibration field {name} must be numeric") from exc


def _as_matrix4(value: Any, *, name: str) -> Matrix4:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"Calibration field {name} must be a 4x4 list")
    matrix: Matrix4 = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"Calibration field {name} must be a 4x4 list")
        matrix.append([_as_float(item, name=name) for item in row])
    return matrix


def _matmul4(a: Matrix4, b: Matrix4) -> Matrix4:
    return [[sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4)] for row in range(4)]


def _invert_rigid4(matrix: Matrix4) -> Matrix4:
    rotation = [[matrix[r][c] for c in range(3)] for r in range(3)]
    translation = [matrix[r][3] for r in range(3)]
    rotation_t = [[rotation[c][r] for c in range(3)] for r in range(3)]
    inv_translation = [-sum(rotation_t[r][c] * translation[c] for c in range(3)) for r in range(3)]
    return [
        [rotation_t[0][0], rotation_t[0][1], rotation_t[0][2], inv_translation[0]],
        [rotation_t[1][0], rotation_t[1][1], rotation_t[1][2], inv_translation[1]],
        [rotation_t[2][0], rotation_t[2][1], rotation_t[2][2], inv_translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _transform_point(matrix: Matrix4, point: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    )


def _first_present(mapping: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def load_aria_projection_calibration(path: str | Path) -> AriaProjectionCalibration:
    """Load a small JSON calibration used for CPF-to-image projection.

    Supported JSON forms:

    1. Direct camera-to-CPF transform:
       {"camera": {"fx": ..., "fy": ..., "cx": ..., "cy": ...},
        "T_camera_cpf": [[...], ...]}

    2. Project Aria-style device transforms:
       {"intrinsics": {"fx": ..., "fy": ..., "cx": ..., "cy": ...},
        "T_device_camera": [[...], ...],
        "T_device_cpf": [[...], ...]}

    The official Project Aria tools path is equivalent to
    inv(T_device_camera) @ T_device_cpf before camera projection.
    Use a VRS or online_calibration.jsonl with projectaria-tools for strict
    native Aria fisheye/Kannala-Brandt camera projection.
    """

    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Calibration JSON must be an object: {path}")
    camera = data.get("camera") or data.get("intrinsics") or data
    if not isinstance(camera, dict):
        raise ValueError(f"Calibration camera/intrinsics must be an object: {path}")

    fx = _as_float(_first_present(camera, ["fx", "f_x"]), name="fx")
    fy = _as_float(_first_present(camera, ["fy", "f_y"]), name="fy")
    cx = _as_float(_first_present(camera, ["cx", "c_x", "ppx"]), name="cx")
    cy = _as_float(_first_present(camera, ["cy", "c_y", "ppy"]), name="cy")
    width_raw = _first_present(camera, ["width", "image_width", "w"])
    height_raw = _first_present(camera, ["height", "image_height", "h"])
    width = int(width_raw) if width_raw is not None else None
    height = int(height_raw) if height_raw is not None else None

    direct = _first_present(
        data,
        [
            "T_camera_cpf",
            "T_camera_CPF",
            "T_rgb_cpf",
            "T_rgb_CPF",
            "transform_camera_cpf",
        ],
    )
    if direct is not None:
        t_camera_cpf = _as_matrix4(direct, name="T_camera_cpf")
    else:
        t_device_camera = _first_present(data, ["T_device_camera", "T_device_rgb", "transform_device_camera"])
        t_device_cpf = _first_present(data, ["T_device_cpf", "T_device_CPF", "transform_device_cpf"])
        if t_device_camera is None or t_device_cpf is None:
            raise ValueError(
                "Calibration JSON must include T_camera_cpf or both T_device_camera and T_device_cpf"
            )
        else:
            t_camera_cpf = _matmul4(
                _invert_rigid4(_as_matrix4(t_device_camera, name="T_device_camera")),
                _as_matrix4(t_device_cpf, name="T_device_cpf"),
            )

    return AriaProjectionCalibration(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        t_camera_cpf=t_camera_cpf,
        source_path=str(path),
    )


def find_clip_calibration(calibration_dir: str | Path | None, clip: dict[str, Any]) -> Path | None:
    if not calibration_dir:
        return None
    root = Path(calibration_dir)
    if root.is_file():
        return root
    clip_id = clip.get("clip_id")
    agent_dir = clip.get("agent_dir")
    day = clip.get("day")
    time_token = clip.get("time_token")
    names = []
    if clip_id:
        names.append(f"{clip_id}.json")
    if day and agent_dir and time_token:
        names.append(f"{day}_{agent_dir}_{time_token}.json")
    if agent_dir and day:
        names.extend(
            [
                f"{agent_dir}_{day}.json",
                f"{agent_dir}_{day}.vrs",
                f"{agent_dir}_{day}_online_calibration.jsonl",
                f"{agent_dir}/{day}.json",
                f"{agent_dir}/{day}.vrs",
                f"{agent_dir}/{day}/online_calibration.jsonl",
            ]
        )
    if agent_dir:
        names.extend(
            [
                f"{agent_dir}.json",
                f"{agent_dir}.vrs",
                f"{agent_dir}/calibration.json",
                f"{agent_dir}/device_calibration.json",
                f"{agent_dir}/online_calibration.jsonl",
            ]
        )
    names.extend(["calibration.json", "device_calibration.json", "online_calibration.jsonl", "recording.vrs"])
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def combined_yaw(row: dict[str, Any]) -> float | None:
    if row.get("yaw_rads_cpf") not in (None, ""):
        return _as_float(row.get("yaw_rads_cpf"), name="yaw_rads_cpf")
    left = row.get("left_yaw_rads_cpf")
    right = row.get("right_yaw_rads_cpf")
    if left in (None, "") or right in (None, ""):
        return None
    return (_as_float(left, name="left_yaw_rads_cpf") + _as_float(right, name="right_yaw_rads_cpf")) / 2.0


def gaze_cpf_point(row: dict[str, Any]) -> tuple[float, float, float] | None:
    """Convert one gaze row from yaw/pitch/depth to a 3D CPF point.

    This follows Project Aria's angle definitions: yaw and pitch are angles
    between the XZ/YZ plane projections and the CPF Z axis. The supplied depth
    is used as the forward CPF Z distance.
    """

    yaw = combined_yaw(row)
    pitch_value = row.get("pitch_rads_cpf")
    depth_value = row.get("depth_m")
    if yaw is None or pitch_value in (None, "") or depth_value in (None, ""):
        return None
    pitch = _as_float(pitch_value, name="pitch_rads_cpf")
    depth = _as_float(depth_value, name="depth_m")
    if not math.isfinite(depth) or depth <= 0:
        return None
    return (math.tan(yaw) * depth, math.tan(pitch) * depth, depth)


def project_gaze_row(
    row: dict[str, Any],
    calibration: AriaProjectionCalibration,
) -> dict[str, Any] | None:
    cpf_point = gaze_cpf_point(row)
    if cpf_point is None:
        return None
    camera_point = _transform_point(calibration.t_camera_cpf, cpf_point)
    x_cam, y_cam, z_cam = camera_point
    if not math.isfinite(z_cam) or z_cam <= 0:
        return None
    x_px = calibration.fx * (x_cam / z_cam) + calibration.cx
    y_px = calibration.fy * (y_cam / z_cam) + calibration.cy
    in_frame = True
    if calibration.width is not None:
        in_frame = in_frame and 0 <= x_px < calibration.width
    if calibration.height is not None:
        in_frame = in_frame and 0 <= y_px < calibration.height
    return {
        "tracking_timestamp_us": row.get("tracking_timestamp_us"),
        "x": round(x_px, 3),
        "y": round(y_px, 3),
        "in_frame": in_frame,
        "cpf_point_m": [round(value, 5) for value in cpf_point],
        "camera_point_m": [round(value, 5) for value in camera_point],
    }


def summarize_projected_gaze(points: list[dict[str, Any]], calibration: AriaProjectionCalibration) -> dict[str, Any]:
    if not points:
        return {
            "projection_status": "no_projectable_rows",
            "projected_point_count": 0,
            "calibration_path": calibration.source_path,
        }
    xs = sorted(float(point["x"]) for point in points)
    ys = sorted(float(point["y"]) for point in points)
    mid = len(points) // 2
    median_x = xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2.0
    median_y = ys[mid] if len(ys) % 2 else (ys[mid - 1] + ys[mid]) / 2.0
    in_frame_count = sum(1 for point in points if point.get("in_frame"))
    return {
        "projection_status": "projected",
        "projection_method": "aria_cpf_depth_to_pinhole_intrinsics",
        "calibration_path": calibration.source_path,
        "projected_point_count": len(points),
        "in_frame_count": in_frame_count,
        "in_frame_ratio": round(in_frame_count / len(points), 4),
        "median_x": round(median_x, 3),
        "median_y": round(median_y, 3),
        "image_width": calibration.width,
        "image_height": calibration.height,
    }


def gaussian_bbox_score(
    gaze_point: tuple[float, float],
    bbox_xyxy: tuple[float, float, float, float],
    *,
    sigma: float,
) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    dx = gaze_point[0] - center_x
    dy = gaze_point[1] - center_y
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return math.exp(-((dx * dx + dy * dy) / (2 * sigma * sigma)))


def project_gaze_csv_with_projectaria_tools(
    gaze_csv_path: str | Path,
    aria_calibration_path: str | Path,
    *,
    max_rows: int = 5000,
    rgb_stream_label: str = "camera-rgb",
    make_upright: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project gaze with official Project Aria camera calibration.

    `aria_calibration_path` may be an Aria VRS/no-image VRS file or an
    MPS `online_calibration.jsonl`. This path uses
    `CameraCalibration.project()` from projectaria-tools, preserving the native
    Aria camera model. It is intentionally optional because the public EgoLife
    HF release currently exposes EyeGaze/EyeTracking files but not calibration
    files in the same tree.
    """

    try:
        from projectaria_tools.core import data_provider, mps
        from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection
    except ImportError as exc:
        raise RuntimeError("projectaria-tools is required for VRS/online-calibration projection") from exc

    calibration_path = Path(aria_calibration_path)
    if calibration_path.suffix == ".jsonl":
        calibrations = mps.read_online_calibration(str(calibration_path))
        if not calibrations:
            raise RuntimeError(f"No online calibration entries found: {calibration_path}")
        device_calibration = calibrations[0]
        stream_label = rgb_stream_label
    else:
        provider = data_provider.create_vrs_data_provider(str(calibration_path))
        device_calibration = provider.get_device_calibration()
        stream_label = rgb_stream_label
        try:
            from projectaria_tools.core.stream_id import StreamId

            stream_label = provider.get_label_from_stream_id(StreamId("214-1")) or stream_label
        except Exception:
            pass
    camera_calibration = device_calibration.get_camera_calib(stream_label)
    if camera_calibration is None:
        raise RuntimeError(f"RGB camera calibration not found for stream label: {stream_label}")

    image_width = None
    image_height = None
    try:
        image_width, image_height = camera_calibration.get_image_size()
    except Exception:
        pass

    points: list[dict[str, Any]] = []
    eye_gazes = mps.read_eyegaze(str(gaze_csv_path))
    for eye_gaze in eye_gazes[:max_rows]:
        depth_m = getattr(eye_gaze, "depth", None) or 1.0
        projected = get_gaze_vector_reprojection(
            eye_gaze,
            stream_label,
            device_calibration,
            camera_calibration,
            depth_m,
            make_upright=make_upright,
        )
        if projected is None:
            continue
        x_px = float(projected[0])
        y_px = float(projected[1])
        in_frame = True
        if image_width is not None:
            in_frame = in_frame and 0 <= x_px < image_width
        if image_height is not None:
            in_frame = in_frame and 0 <= y_px < image_height
        points.append(
            {
                "tracking_timestamp_us": int(eye_gaze.tracking_timestamp.total_seconds() * 1_000_000),
                "x": round(x_px, 3),
                "y": round(y_px, 3),
                "in_frame": in_frame,
            }
        )

    summary = {
        "projection_status": "projected" if points else "no_projectable_rows",
        "projection_method": "projectaria_tools_camera_calibration_project",
        "calibration_path": str(calibration_path),
        "rgb_stream_label": stream_label,
        "projected_point_count": len(points),
        "image_width": image_width,
        "image_height": image_height,
    }
    if points:
        xs = sorted(float(point["x"]) for point in points)
        ys = sorted(float(point["y"]) for point in points)
        mid = len(points) // 2
        summary.update(
            {
                "in_frame_count": sum(1 for point in points if point.get("in_frame")),
                "median_x": round(xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2.0, 3),
                "median_y": round(ys[mid] if len(ys) % 2 else (ys[mid - 1] + ys[mid]) / 2.0, 3),
            }
        )
    return points, summary
