#!/usr/bin/env python3
"""Incrementally inject depth_path metadata into ann PKL camera entries.

This script is designed for TartanGround-style ann files and keeps the input
container format unchanged. It only adds/updates optional `depth_path` fields
under `cams` and `cam_sweeps` camera dicts.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class UpdateStats:
    total_infos: int = 0
    total_cam_entries: int = 0
    added_count: int = 0
    overwritten_count: int = 0
    existed_count: int = 0
    missing_file_count: int = 0
    invalid_cam_info_count: int = 0
    strict_failure_count: int = 0

    def to_summary(self) -> Dict[str, Any]:
        depth_exists_count = self.total_cam_entries - self.missing_file_count
        exists_ratio = (
            float(depth_exists_count) / float(self.total_cam_entries)
            if self.total_cam_entries > 0
            else 1.0
        )
        return {
            **asdict(self),
            "depth_exists_count": depth_exists_count,
            "depth_exists_ratio": exists_ratio,
        }


def _load_container(path: Path) -> Tuple[Any, List[Dict[str, Any]], Optional[str]]:
    with path.open("rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        if isinstance(data.get("infos"), list):
            return data, data["infos"], "infos"
        if isinstance(data.get("data_list"), list):
            return data, data["data_list"], "data_list"
        raise TypeError(
            f"Unsupported dict ann format in {path}: expected key 'infos' or 'data_list' as list"
        )

    if isinstance(data, list):
        return data, data, None

    raise TypeError(f"Unsupported ann object type in {path}: {type(data)!r}")


def _atomic_dump_pickle(obj: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, out_path)


def _map_image_to_depth_candidates(image_path: str) -> Tuple[str, str]:
    image_path = str(image_path)
    p = Path(image_path)

    parent_name = p.parent.name
    if parent_name.startswith("image_"):
        depth_dir = p.parent.parent / parent_name.replace("image_", "depth_", 1)
    else:
        depth_dir = p.parent

    stem = p.stem
    if not stem.endswith("_depth"):
        depth_stem = stem + "_depth"
    else:
        depth_stem = stem

    cand_png = depth_dir / f"{depth_stem}.png"
    cand_npy = depth_dir / f"{depth_stem}.npy"
    return str(cand_png), str(cand_npy)


def _resolve_depth_path(image_path: str) -> Tuple[str, str, str, bool]:
    cand_png, cand_npy = _map_image_to_depth_candidates(image_path)
    if os.path.exists(cand_png):
        return cand_png, cand_png, cand_npy, True
    if os.path.exists(cand_npy):
        return cand_npy, cand_png, cand_npy, True
    return cand_png, cand_png, cand_npy, False


def _iter_sweep_groups(cam_sweeps: Any) -> Iterable[Tuple[str, Any]]:
    # TartanGround ann style: list[dict[cam_name -> cam_info]]
    if isinstance(cam_sweeps, list):
        for sweep_idx, sweep in enumerate(cam_sweeps):
            yield f"cam_sweeps[{sweep_idx}]", sweep
        return

    # Alternative style: {'prev': [...], 'next': [...]}.
    if isinstance(cam_sweeps, dict):
        for side in ["prev", "next"]:
            sweeps = cam_sweeps.get(side, [])
            if not isinstance(sweeps, list):
                continue
            for sweep_idx, sweep in enumerate(sweeps):
                yield f"cam_sweeps.{side}[{sweep_idx}]", sweep


def _record_missing(
    missing_examples: List[Dict[str, Any]],
    max_missing_examples: int,
    *,
    token: str,
    scope: str,
    cam_name: str,
    image_path: str,
    candidate_png: str,
    candidate_npy: str,
    reason: str,
) -> None:
    if len(missing_examples) >= max_missing_examples:
        return
    missing_examples.append(
        {
            "token": token,
            "scope": scope,
            "cam_name": cam_name,
            "image_path": image_path,
            "candidate_png": candidate_png,
            "candidate_npy": candidate_npy,
            "reason": reason,
        }
    )


def _update_cam_info_depth_path(
    cam_info: Dict[str, Any],
    *,
    token: str,
    scope: str,
    cam_name: str,
    stats: UpdateStats,
    missing_examples: List[Dict[str, Any]],
    skip_if_exists: bool,
    strict_exists: bool,
    max_missing_examples: int,
) -> None:
    stats.total_cam_entries += 1

    existing_depth_path = cam_info.get("depth_path")
    if existing_depth_path and skip_if_exists:
        stats.existed_count += 1
        return

    image_path = cam_info.get("data_path")
    if not isinstance(image_path, str) or not image_path:
        stats.invalid_cam_info_count += 1
        stats.missing_file_count += 1
        _record_missing(
            missing_examples,
            max_missing_examples,
            token=token,
            scope=scope,
            cam_name=cam_name,
            image_path=str(image_path),
            candidate_png="",
            candidate_npy="",
            reason="invalid_or_missing_data_path",
        )
        if strict_exists:
            stats.strict_failure_count += 1
        return

    depth_path, cand_png, cand_npy, depth_exists = _resolve_depth_path(image_path)

    if existing_depth_path and not skip_if_exists:
        stats.overwritten_count += 1
    else:
        stats.added_count += 1

    cam_info["depth_path"] = depth_path

    if not depth_exists:
        stats.missing_file_count += 1
        _record_missing(
            missing_examples,
            max_missing_examples,
            token=token,
            scope=scope,
            cam_name=cam_name,
            image_path=image_path,
            candidate_png=cand_png,
            candidate_npy=cand_npy,
            reason="depth_file_not_found",
        )
        if strict_exists:
            stats.strict_failure_count += 1


def _check_depth_path_structure(
    infos: Sequence[Dict[str, Any]],
    *,
    sample_count: int,
    seed: int,
) -> Dict[str, Any]:
    if not infos:
        return {
            "checked_samples": 0,
            "checked_indices": [],
            "missing_depth_key_count": 0,
            "missing_examples": [],
            "passed": True,
        }

    sample_count = max(1, min(sample_count, len(infos)))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(infos)), sample_count))

    missing_examples: List[Dict[str, Any]] = []
    missing_depth_key_count = 0

    def _check_group(group: Any, *, token: str, scope: str) -> None:
        nonlocal missing_depth_key_count
        if not isinstance(group, dict):
            return
        for cam_name, cam_info in group.items():
            if not isinstance(cam_info, dict):
                continue
            depth_path = cam_info.get("depth_path")
            if not isinstance(depth_path, str) or not depth_path:
                missing_depth_key_count += 1
                if len(missing_examples) < 50:
                    missing_examples.append(
                        {
                            "token": token,
                            "scope": scope,
                            "cam_name": cam_name,
                            "reason": "missing_or_invalid_depth_path",
                        }
                    )

    for idx in indices:
        info = infos[idx]
        token = str(info.get("token", f"index_{idx}"))
        _check_group(info.get("cams"), token=token, scope="cams")
        for scope, sweep in _iter_sweep_groups(info.get("cam_sweeps")):
            _check_group(sweep, token=token, scope=scope)

    return {
        "checked_samples": sample_count,
        "checked_indices": indices,
        "missing_depth_key_count": missing_depth_key_count,
        "missing_examples": missing_examples,
        "passed": missing_depth_key_count == 0,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add depth_path to cams/cam_sweeps in ann pkl without modifying source file."
    )
    parser.add_argument("--in-pkl", required=True, help="Input ann pkl path")
    parser.add_argument("--out-pkl", required=True, help="Output ann pkl path")
    parser.add_argument(
        "--strict-exists",
        action="store_true",
        default=False,
        help="Fail if any mapped depth file does not exist.",
    )
    parser.add_argument(
        "--skip-if-exists",
        dest="skip_if_exists",
        action="store_true",
        default=True,
        help="Do not overwrite existing depth_path fields (default: true).",
    )
    parser.add_argument(
        "--no-skip-if-exists",
        dest="skip_if_exists",
        action="store_false",
        help="Overwrite existing depth_path fields.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional report json output path.",
    )
    parser.add_argument(
        "--max-missing-report",
        type=int,
        default=200,
        help="Maximum number of missing entries written into report.",
    )
    parser.add_argument(
        "--sample-checks",
        type=int,
        default=5,
        help="How many random samples to verify depth_path structure.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=2026,
        help="Random seed used for sample structure checks.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()

    in_path = Path(args.in_pkl)
    out_path = Path(args.out_pkl)
    report_path = Path(args.report_json) if args.report_json else None

    if not in_path.exists():
        raise FileNotFoundError(f"Input ann file not found: {in_path}")

    if in_path.resolve() == out_path.resolve():
        raise ValueError("Input and output ann pkl must be different paths to protect original data.")

    container, infos, container_key = _load_container(in_path)
    stats = UpdateStats(total_infos=len(infos))
    missing_examples: List[Dict[str, Any]] = []

    for idx, info in enumerate(infos):
        if not isinstance(info, dict):
            continue
        token = str(info.get("token", f"index_{idx}"))

        cams = info.get("cams")
        if isinstance(cams, dict):
            for cam_name, cam_info in cams.items():
                if not isinstance(cam_info, dict):
                    continue
                _update_cam_info_depth_path(
                    cam_info,
                    token=token,
                    scope="cams",
                    cam_name=str(cam_name),
                    stats=stats,
                    missing_examples=missing_examples,
                    skip_if_exists=args.skip_if_exists,
                    strict_exists=args.strict_exists,
                    max_missing_examples=args.max_missing_report,
                )

        for sweep_scope, sweep in _iter_sweep_groups(info.get("cam_sweeps")):
            if not isinstance(sweep, dict):
                continue
            for cam_name, cam_info in sweep.items():
                if not isinstance(cam_info, dict):
                    continue
                _update_cam_info_depth_path(
                    cam_info,
                    token=token,
                    scope=sweep_scope,
                    cam_name=str(cam_name),
                    stats=stats,
                    missing_examples=missing_examples,
                    skip_if_exists=args.skip_if_exists,
                    strict_exists=args.strict_exists,
                    max_missing_examples=args.max_missing_report,
                )

    if args.strict_exists and stats.strict_failure_count > 0:
        raise RuntimeError(
            f"Strict exists check failed: {stats.strict_failure_count} missing/invalid depth entries."
        )

    _atomic_dump_pickle(container, out_path)

    sample_check = _check_depth_path_structure(
        infos,
        sample_count=args.sample_checks,
        seed=args.sample_seed,
    )

    summary = stats.to_summary()
    report = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "in_pkl": str(in_path),
        "out_pkl": str(out_path),
        "container_key": container_key,
        "strict_exists": bool(args.strict_exists),
        "skip_if_exists": bool(args.skip_if_exists),
        "summary": summary,
        "sample_structure_check": sample_check,
        "missing_examples": missing_examples,
    }

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print("[add_depth_paths_to_ann] Done")
    print(f"  in_pkl: {in_path}")
    print(f"  out_pkl: {out_path}")
    print(f"  total_infos: {summary['total_infos']}")
    print(f"  total_cam_entries: {summary['total_cam_entries']}")
    print(f"  added_count: {summary['added_count']}")
    print(f"  overwritten_count: {summary['overwritten_count']}")
    print(f"  existed_count: {summary['existed_count']}")
    print(f"  missing_file_count: {summary['missing_file_count']}")
    print(f"  depth_exists_ratio: {summary['depth_exists_ratio']:.6f}")
    print(
        "  sample_structure_check: "
        f"checked={sample_check['checked_samples']} "
        f"missing_depth_key_count={sample_check['missing_depth_key_count']}"
    )
    if report_path is not None:
        print(f"  report_json: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
