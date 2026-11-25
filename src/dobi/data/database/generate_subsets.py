# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
import json
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Set


_METRICS_KEYS = {
    "class_id",
    "min_gt_count",
    "max_gt_count",
    "min_box_size",
    "max_box_size",
    "min_iou",
    "score_threshold",
}


def _metrics_filter_is_active(metrics_filter: Dict[str, Any]) -> bool:
    if not isinstance(metrics_filter, dict) or not metrics_filter:
        return False
    return any(key in metrics_filter and metrics_filter[key] is not None for key in _METRICS_KEYS)


def build_subset_from_params(
    db_path: str,
    name: str,
    params: Dict[str, Any],
    split: str,
    replace_existing: bool = False,
) -> int:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    cur = conn.cursor()

    if replace_existing:
        cur.execute(
            "DELETE FROM subsets WHERE name = ? AND split = ?",
            (name, split),
        )

    params_json = json.dumps(params, sort_keys=True)
    created_at = datetime.now(timezone.utc).isoformat()

    cur.execute(
        """
        INSERT INTO subsets (name, split, params_json, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (name, split, params_json, created_at),
    )
    subset_id = cur.lastrowid

    degradation_filter = params.get("degradation_filter", {})
    metrics_filter = params.get("metrics_filter", {})
    manual_additions = params.get("manual_additions", [])

    include_specs = degradation_filter.get("include", [])

    use_metrics = _metrics_filter_is_active(metrics_filter)
    class_id = metrics_filter.get("class_id")

    if use_metrics and class_id is None:
        raise ValueError(
            f"Subset '{name}' uses metrics_filter but does not define 'class_id'."
        )

    min_gt_count = metrics_filter.get("min_gt_count")
    max_gt_count = metrics_filter.get("max_gt_count")
    min_box_size = metrics_filter.get("min_box_size")
    max_box_size = metrics_filter.get("max_box_size")
    min_iou = metrics_filter.get("min_iou")
    score_threshold = metrics_filter.get("score_threshold", 0.0)

    selected_variant_ids: Set[int] = set()

    base_sql = """
        SELECT v.id, v.image_id, v.deg_name, v.severity
        FROM variants v
        JOIN images i ON v.image_id = i.id
    """
    params_sql: List[Any] = [split]

    if include_specs:
        where_clauses = []
        for spec in include_specs:
            deg_name = spec.get("deg_name")
            if deg_name is None:
                raise ValueError(f"Subset '{name}' has include spec without 'deg_name': {spec}")

            smin = spec.get("severity_min", 0)
            smax = spec.get("severity_max", 0)

            where_clauses.append("(v.deg_name = ? AND v.severity BETWEEN ? AND ?)")
            params_sql.extend([deg_name, smin, smax])

        where_sql = "WHERE i.split = ? AND (" + " OR ".join(where_clauses) + ")"
    else:
        where_sql = "WHERE i.split = ?"

    cur.execute(base_sql + " " + where_sql, params_sql)
    candidates = cur.fetchall()

    print(
        f"[subset '{name}' split='{split}'] "
        f"#candidate variants from degradation filter: {len(candidates)}"
    )

    count_after_gt = 0
    count_after_size = 0
    count_after_iou = 0

    for variant_id, image_id, deg_name, severity in candidates:
        if not use_metrics:
            selected_variant_ids.add(variant_id)
            continue

        cur.execute(
            """
            SELECT x, y, w, h
            FROM gt_objects
            WHERE image_id = ? AND class_id = ?
            """,
            (image_id, class_id),
        )
        gt_boxes = cur.fetchall()
        gt_count = len(gt_boxes)

        if min_gt_count is not None and gt_count < min_gt_count:
            continue
        if max_gt_count is not None and gt_count > max_gt_count:
            continue
        count_after_gt += 1

        if gt_boxes:
            sizes = [min(w, h) for (_, _, w, h) in gt_boxes]
            min_size_val = min(sizes)
            max_size_val = max(sizes)
        else:
            min_size_val = None
            max_size_val = None

        if min_box_size is not None and (min_size_val is None or min_size_val < min_box_size):
            continue
        if max_box_size is not None and (max_size_val is None or max_size_val > max_box_size):
            continue
        count_after_size += 1

        if min_iou is not None:
            cur.execute(
                """
                SELECT max_iou
                FROM predictions
                WHERE variant_id = ?
                  AND category_id = ?
                  AND score >= ?
                """,
                (variant_id, class_id, score_threshold),
            )
            rows = cur.fetchall()
            if not rows:
                continue

            max_iou_for_variant = max(
                (r[0] for r in rows if r[0] is not None),
                default=None,
            )
            if max_iou_for_variant is None or max_iou_for_variant < min_iou:
                continue
            count_after_iou += 1

        selected_variant_ids.add(variant_id)

    print(f"[subset '{name}' split='{split}'] after GT-count filter : {count_after_gt}")
    print(f"[subset '{name}' split='{split}'] after size filter     : {count_after_size}")
    if min_iou is not None:
        print(f"[subset '{name}' split='{split}'] after IoU filter      : {count_after_iou}")

    manual_added = 0
    for item in manual_additions:
        if "variant_id" in item:
            vid = int(item["variant_id"])
            cur.execute(
                """
                SELECT v.id
                FROM variants v
                JOIN images i ON v.image_id = i.id
                WHERE v.id = ? AND i.split = ?
                """,
                (vid, split),
            )
            row = cur.fetchone()
            if row:
                selected_variant_ids.add(row[0])
                manual_added += 1
            continue

        image_id_m = item.get("image_id")
        deg_name_m = item.get("deg_name")
        severity_m = item.get("severity")
        if image_id_m is not None and deg_name_m is not None and severity_m is not None:
            cur.execute(
                """
                SELECT v.id
                FROM variants v
                JOIN images i ON v.image_id = i.id
                WHERE i.id = ?
                  AND i.split = ?
                  AND v.deg_name = ?
                  AND v.severity = ?
                """,
                (image_id_m, split, deg_name_m, severity_m),
            )
            row = cur.fetchone()
            if row:
                selected_variant_ids.add(row[0])
                manual_added += 1

    print(f"[subset '{name}' split='{split}'] manual additions      : {manual_added}")

    selected_variant_ids_list = sorted(selected_variant_ids)
    cur.executemany(
        """
        INSERT INTO subset_variants (subset_id, variant_id)
        VALUES (?, ?)
        """,
        [(subset_id, vid) for vid in selected_variant_ids_list],
    )

    conn.commit()
    conn.close()

    print(
        f"[subset '{name}' split='{split}'] total selected variants: "
        f"{len(selected_variant_ids_list)}"
    )
    print(f"[subset '{name}' split='{split}'] stored with id={subset_id}")
    return subset_id


def build_subsets_from_json(
    db_path: str,
    config_path: str,
    split: str,
    replace_existing: bool = False,
) -> Dict[str, int]:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    subsets_cfg = cfg.get("subsets")
    if subsets_cfg is None:
        subsets_cfg = cfg.get("clusters", [])

    if not isinstance(subsets_cfg, list) or not subsets_cfg:
        raise ValueError(f"No valid 'subsets' list found in {config_path}")

    created: Dict[str, int] = {}

    print(f"Building {len(subsets_cfg)} subsets from {config_path} for split='{split}'")

    for idx, item in enumerate(subsets_cfg):
        name = item.get("name")
        params = item.get("params")

        if not name or not isinstance(params, dict):
            print(f"  [skip] entry {idx} has invalid 'name' or 'params'")
            continue

        print(f"\n--- Creating subset {idx}: '{name}' (split='{split}') ---")
        subset_id = build_subset_from_params(
            db_path=db_path,
            name=name,
            params=params,
            split=split,
            replace_existing=replace_existing,
        )
        created[f"{name}_{split}"] = subset_id

    print("\nAll requested subsets processed.")
    return created


def main():
    parser = argparse.ArgumentParser(
        description="Build split-specific subsets into the SQLite DB."
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to SQLite DB file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config with subset definitions.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "test", "val"],
        help="Splits to build subsets for (e.g. --splits train val).",
    )
    parser.add_argument(
        "--replace_existing",
        action="store_false",
        help="Delete existing subsets with the same (name, split) before rebuilding.",
    )

    args = parser.parse_args()

    for split in args.splits:
        print(f"\n=== Building subsets for split='{split}' ===")
        build_subsets_from_json(
            args.db,
            args.config,
            split,
            replace_existing=args.replace_existing,
        )


if __name__ == "__main__":
    main()