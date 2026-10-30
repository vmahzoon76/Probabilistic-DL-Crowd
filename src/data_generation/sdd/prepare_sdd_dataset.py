import argparse
import json
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


DT = 0.4
DOWNSAMPLE = 12
RAW_FPS = 30.0
LABEL = "Pedestrian"
EXCLUDED_SCALE_VIDEOS = {("hyang", "video13"), ("hyang", "video14")}
CONSTRAINED_CLASSES = ("Building", "Obstacle", "Object", "Offroad")
PHYSICAL_OBSTACLE_CLASSES = ("Building", "Obstacle", "Object")
OUTPUT_COLS = ["pedestrian_ID", "pred_vel_x", "pred_vel_y"]
INPUT_COLS = [
    "pedestrian_ID",
    "curr_loc_x",
    "curr_loc_y",
    "curr_vel_x",
    "curr_vel_y",
    "goal_loc_x",
    "goal_loc_y",
]


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def scan_lidar(
    polygons: list[list[list[float]]],
    agents: list[list[float]],
    scan_range: float = 10.0,
    angle_resolution: float = 0.5,
    agent_radius: float = 0.2,
) -> list[float]:
    angles = np.linspace(0, 2.0 * math.pi, int(360 / angle_resolution), False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    scan = np.full(len(angles), scan_range, dtype=np.float64)
    eps = 1e-9

    p1_list = []
    seg_list = []
    for polygon in polygons:
        if len(polygon) < 2:
            continue
        vertices = np.array(polygon, dtype=np.float64)
        p1_list.append(vertices)
        seg_list.append(np.roll(vertices, -1, axis=0) - vertices)
    if p1_list:
        p1 = np.vstack(p1_list)
        seg = np.vstack(seg_list)
        denom = directions[:, None, 0] * seg[None, :, 1] - directions[:, None, 1] * seg[None, :, 0]
        numerator_t = p1[None, :, 0] * seg[None, :, 1] - p1[None, :, 1] * seg[None, :, 0]
        numerator_u = p1[None, :, 0] * directions[:, None, 1] - p1[None, :, 1] * directions[:, None, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = numerator_t / denom
            u = numerator_u / denom
        valid = (np.abs(denom) > eps) & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
        t_valid = np.where(valid, t, scan_range)
        scan = np.minimum(scan, np.min(t_valid, axis=1))

    if agents:
        centers = np.array(agents, dtype=np.float64)
        projection = directions @ centers.T
        center_norm_sq = np.sum(centers * centers, axis=1)
        closest_sq = center_norm_sq[None, :] - projection * projection
        radius_sq = agent_radius * agent_radius
        hit = (closest_sq <= radius_sq) & (projection >= -agent_radius)
        offset = np.sqrt(np.maximum(radius_sq - closest_sq, 0.0))
        near = projection - offset
        far = projection + offset
        candidates = np.where(near >= 0.0, near, far)
        candidates = np.where(hit & (candidates >= 0.0), candidates, scan_range)
        scan = np.minimum(scan, np.min(candidates, axis=1))

    return scan.tolist()


def natural_video_key(path: Path) -> tuple[str, int]:
    match = re.fullmatch(r"video(\d+)", path.name)
    if not match:
        raise ValueError(f"Unexpected video folder name: {path}")
    return path.parent.name, int(match.group(1))


def constrained_video_key(path: Path) -> tuple[str, str]:
    return path.parent.name, path.name


def load_scales(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    with path.open() as handle:
        raw = yaml.safe_load(handle)
    scales = {}
    for scene, videos in raw.items():
        for video, values in videos.items():
            scales[(scene, video)] = {
                "scale": float(values["scale"]),
                "certainty": float(values["certainty"]),
            }
    return scales


def read_annotation(path: Path) -> pd.DataFrame:
    names = [
        "track_id",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "frame",
        "lost",
        "occluded",
        "generated",
        "label",
    ]
    df = pd.read_csv(path, sep=r"\s+", header=None, quotechar='"', names=names)
    df["label"] = df["label"].astype(str).str.strip('"')
    for col in names[:-1]:
        df[col] = pd.to_numeric(df[col])
    return df


def velocity_for_positions(points: np.ndarray) -> np.ndarray:
    if len(points) == 1:
        return np.zeros((1, 2), dtype=np.float64)
    vel = np.zeros_like(points, dtype=np.float64)
    vel[0] = (points[1] - points[0]) / DT
    vel[-1] = (points[-1] - points[-2]) / DT
    if len(points) > 2:
        vel[1:-1] = (points[2:] - points[:-2]) / (2.0 * DT)
    return vel


def split_contiguous(track_df: pd.DataFrame) -> list[pd.DataFrame]:
    track_df = track_df.sort_values("frame").reset_index(drop=True)
    if track_df.empty:
        return []
    frame_steps = track_df["frame"].diff().fillna(DOWNSAMPLE)
    split_at = list(np.where(frame_steps.to_numpy() != DOWNSAMPLE)[0])
    starts = [0] + split_at
    ends = split_at + [len(track_df)]
    return [track_df.iloc[start:end].copy() for start, end in zip(starts, ends) if end > start]


def make_goal_table(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["pedestrian_ID", "First_Frame", "Last_Frame", "g_x", "g_y"])
    return pd.DataFrame(rows).sort_values("pedestrian_ID").reset_index(drop=True)


def frame_dicts_from_obsmat(df: pd.DataFrame, min_frame: int, max_frame: int) -> list[dict[int, np.ndarray]]:
    out = []
    grouped = {frame: rows for frame, rows in df.groupby("frame_number")}
    for frame in range(min_frame, max_frame + 1):
        frame_dict = {}
        rows = grouped.get(frame)
        if rows is not None:
            for row in rows.itertuples(index=False):
                frame_dict[int(row.pedestrian_ID)] = np.array(
                    [row.pos_x, row.pos_y, row.v_x, row.v_y], dtype=np.float64
                )
        out.append(frame_dict)
    return out


def build_transformer_samples(df: pd.DataFrame, goals: pd.DataFrame) -> list[list[pd.DataFrame]]:
    goal_by_id = {
        int(row.pedestrian_ID): (float(row.g_x), float(row.g_y))
        for row in goals.itertuples(index=False)
    }
    frames = {frame: rows for frame, rows in df.groupby("frame_number")}
    samples = []
    for frame in sorted(frames):
        next_rows = frames.get(frame + 1)
        if next_rows is None:
            continue
        current = frames[frame]
        next_by_ped = {int(row.pedestrian_ID): row for row in next_rows.itertuples(index=False)}
        for target in current.itertuples(index=False):
            ped = int(target.pedestrian_ID)
            if ped not in next_by_ped:
                continue
            ordered_rows = [target] + [
                row for row in current.itertuples(index=False) if int(row.pedestrian_ID) != ped
            ]
            input_rows = []
            for row in ordered_rows:
                row_ped = int(row.pedestrian_ID)
                gx, gy = goal_by_id[row_ped]
                input_rows.append(
                    [
                        row_ped,
                        float(row.pos_x),
                        float(row.pos_y),
                        float(row.v_x),
                        float(row.v_y),
                        gx,
                        gy,
                    ]
                )
            input_df = pd.DataFrame(input_rows, columns=INPUT_COLS)
            nxt = next_by_ped[ped]
            output_df = pd.DataFrame(
                [[ped, float(nxt.v_x), float(nxt.v_y)]],
                columns=OUTPUT_COLS,
            )
            samples.append([input_df, output_df])
    return samples


def make_lidar_vector(input_df: pd.DataFrame, environment_m: list[list[list[float]]]) -> list[float]:
    curr_loc_df = input_df[["curr_loc_x", "curr_loc_y"]]
    curr_vel_df = input_df[["curr_vel_x", "curr_vel_y"]]
    goal_df = input_df[["goal_loc_x", "goal_loc_y"]]

    pos = np.array(curr_loc_df.iloc[0], dtype=np.float64)
    goal_rel = (np.array(goal_df.iloc[0], dtype=np.float64) - pos).tolist()
    curr_vel = np.array(curr_vel_df.iloc[0], dtype=np.float64).tolist()

    curr_rel = np.array(curr_loc_df, dtype=np.float64) - pos
    curr_agents = curr_rel.tolist()[1:]

    prev_center = pos - DT * np.array(curr_vel_df.iloc[0], dtype=np.float64)
    prev_rel = (
        np.array(curr_loc_df, dtype=np.float64)
        - DT * np.array(curr_vel_df, dtype=np.float64)
        - prev_center
    )
    prev_agents = prev_rel.tolist()[1:]

    curr_obs = [(np.array(poly, dtype=np.float64) - pos).tolist() for poly in environment_m]
    prev_obs = [(np.array(poly, dtype=np.float64) - prev_center).tolist() for poly in environment_m]

    curr_scan = scan_lidar(curr_obs, curr_agents)
    prev_scan = scan_lidar(prev_obs, prev_agents)
    return prev_scan + curr_scan + curr_vel + goal_rel


def build_lidar_samples(
    transformer_samples: list[list[pd.DataFrame]],
    environment_m: list[list[list[float]]],
) -> list[list[list[float]]]:
    data = [[], []]
    for input_df, output_df in transformer_samples:
        data[0].append(make_lidar_vector(input_df, environment_m))
        data[1].append(
            np.array(output_df[["pred_vel_x", "pred_vel_y"]], dtype=np.float64).reshape(2).tolist()
        )
    return data


def candidate_id_switches(segments: list[dict], max_gap: int = 2, max_dist_m: float = 0.75) -> list[dict]:
    by_video = defaultdict(list)
    for segment in segments:
        by_video[(segment["scene"], segment["video"])].append(segment)

    candidates = []
    for (scene, video), items in by_video.items():
        endings = sorted(items, key=lambda item: item["last_frame"])
        starts = sorted(items, key=lambda item: item["first_frame"])
        for end_item in endings:
            for start_item in starts:
                if end_item["pedestrian_ID"] == start_item["pedestrian_ID"]:
                    continue
                gap = start_item["first_frame"] - end_item["last_frame"]
                if gap <= 0 or gap > max_gap:
                    continue
                dist = math.dist(end_item["end_pos"], start_item["start_pos"])
                if dist <= max_dist_m:
                    candidates.append(
                        {
                            "scene": scene,
                            "video": video,
                            "from_pedestrian_ID": end_item["pedestrian_ID"],
                            "to_pedestrian_ID": start_item["pedestrian_ID"],
                            "gap_frames_2p5fps": gap,
                            "distance_m": dist,
                        }
                    )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="Datasets/SDD")
    parser.add_argument("--skip-lidar", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    sources = dataset_root / "_sources"
    sdd_raw = sources / "StanfordDroneDataset"
    scales_path = sources / "OpenTraj" / "estimated_scales.yaml"
    polygons_path = sources / "constrained-sdd-data" / "polygons.pkl"
    shapes_path = sources / "constrained-sdd-data" / "image_shapes.json"

    scales = load_scales(scales_path)
    with polygons_path.open("rb") as handle:
        constrained_polygons = pickle.load(handle)
    with shapes_path.open() as handle:
        constrained_shapes = {int(k): tuple(v) for k, v in json.load(handle).items()}

    annotation_paths = sorted(
        sdd_raw.glob("*/*/annotations.txt"),
        key=lambda path: constrained_video_key(path.parent),
    )

    img_to_video = {}
    for img_id, ann_path in enumerate(annotation_paths):
        scene, video = ann_path.parts[-3], ann_path.parts[-2]
        if img_id in constrained_polygons:
            img_to_video[img_id] = (scene, video)

    videos_with_polygons = {
        value: img_id for img_id, value in img_to_video.items() if value not in EXCLUDED_SCALE_VIDEOS
    }

    processed_root = dataset_root / "processed"
    transformer_root = dataset_root / "data_for_model" / "transformer"
    lidar_root = dataset_root / "data_for_model" / "lidar"
    processed_root.mkdir(parents=True, exist_ok=True)
    transformer_root.mkdir(parents=True, exist_ok=True)
    lidar_root.mkdir(parents=True, exist_ok=True)

    all_transformer = []
    all_lidar = [[], []]
    all_obsmat = []
    all_goal_tables = []
    all_gt = []
    all_segments = []
    all_samples_by_video = {}
    all_lidar_by_video = {}
    report = {
        "dt_seconds": DT,
        "raw_fps": RAW_FPS,
        "downsample_every_n_frames": DOWNSAMPLE,
        "label_kept": LABEL,
        "excluded_scale_videos": [list(item) for item in sorted(EXCLUDED_SCALE_VIDEOS)],
        "constrained_mapping_order": "scene lexicographic, video folder lexicographic",
        "videos": {},
    }

    next_global_ped = 1
    next_global_frame = 0
    split_events = []
    jump_events = []

    for ann_path in annotation_paths:
        scene, video = ann_path.parts[-3], ann_path.parts[-2]
        key = (scene, video)
        print(f"processing {scene}/{video}", flush=True)
        if key not in videos_with_polygons:
            reason = "missing_constrained_polygons"
            if key in EXCLUDED_SCALE_VIDEOS:
                reason = "excluded_zero_scale_certainty"
            report["videos"][f"{scene}/{video}"] = {"status": "excluded", "reason": reason}
            continue

        scale_info = scales.get(key)
        if scale_info is None:
            report["videos"][f"{scene}/{video}"] = {"status": "excluded", "reason": "missing_opentraj_scale"}
            continue
        if scale_info["certainty"] <= 0:
            report["videos"][f"{scene}/{video}"] = {
                "status": "excluded",
                "reason": "non_positive_opentraj_scale_certainty",
            }
            continue

        img_id = videos_with_polygons[key]
        shape_h, shape_w = constrained_shapes[img_id]
        polygon_resize_to_raw = 5.0
        scale = scale_info["scale"]

        env_m = []
        env_by_class_m = {cls: [] for cls in CONSTRAINED_CLASSES}
        polygon_class_counts = {}
        for cls in CONSTRAINED_CLASSES:
            polys = constrained_polygons[img_id].get(cls, [])
            if cls == "Object":
                polygon_class_counts["Obstacle_alias_Object"] = len(polys)
            else:
                polygon_class_counts[cls] = len(polys)
            for poly in polys:
                raw_px = np.array(poly, dtype=np.float64) * polygon_resize_to_raw
                poly_m = (raw_px * scale).tolist()
                env_m.append(poly_m)
                env_by_class_m[cls].append(poly_m)

        raw = read_annotation(ann_path)
        raw_ped = raw[raw["label"] == LABEL].copy()
        valid = raw_ped[(raw_ped["lost"] == 0) & (raw_ped["frame"] % DOWNSAMPLE == 0)].copy()
        valid["pos_x"] = ((valid["xmin"] + valid["xmax"]) * 0.5) * scale
        valid["pos_y"] = ((valid["ymin"] + valid["ymax"]) * 0.5) * scale
        valid["downsample_frame"] = (valid["frame"] // DOWNSAMPLE).astype(int)
        video_frame_zero = int(valid["downsample_frame"].min()) if not valid.empty else 0

        out_rows = []
        goal_rows = []
        video_segments = []
        local_split_count = 0
        local_short_segments = 0

        for track_id, track in valid.groupby("track_id"):
            parts = split_contiguous(track)
            if len(parts) > 1:
                split_events.append(
                    {
                        "scene": scene,
                        "video": video,
                        "track_id": int(track_id),
                        "segments": len(parts),
                        "reason": "internal_missing_downsampled_valid_frames",
                    }
                )
                local_split_count += 1

            for segment_idx, segment in enumerate(parts):
                if len(segment) < 2:
                    local_short_segments += 1
                    continue
                ped_id = next_global_ped
                next_global_ped += 1
                frames = segment["downsample_frame"].to_numpy(dtype=int)
                local_frames = frames - video_frame_zero
                points = segment[["pos_x", "pos_y"]].to_numpy(dtype=np.float64)
                velocities = velocity_for_positions(points)

                speed = np.linalg.norm(velocities, axis=1)
                if np.nanmax(speed) > 5.0:
                    jump_events.append(
                        {
                            "scene": scene,
                            "video": video,
                            "track_id": int(track_id),
                            "pedestrian_ID": ped_id,
                            "max_speed_m_s": float(np.nanmax(speed)),
                        }
                    )

                goal = points[-1]
                for idx, row in enumerate(segment.itertuples(index=False)):
                    out_rows.append(
                        {
                            "frame_number": int(local_frames[idx]),
                            "pedestrian_ID": ped_id,
                            "pos_x": float(points[idx, 0]),
                            "pos_z": 0.0,
                            "pos_y": float(points[idx, 1]),
                            "v_x": float(velocities[idx, 0]),
                            "v_z": 0.0,
                            "v_y": float(velocities[idx, 1]),
                            "raw_track_id": int(track_id),
                            "raw_frame": int(row.frame),
                            "occluded": int(row.occluded),
                            "generated": int(row.generated),
                        }
                    )
                goal_rows.append(
                    {
                        "pedestrian_ID": ped_id,
                        "First_Frame": int(local_frames[0]),
                        "Last_Frame": int(local_frames[-1]),
                        "g_x": float(goal[0]),
                        "g_y": float(goal[1]),
                        "scene": scene,
                        "video": video,
                        "raw_track_id": int(track_id),
                        "segment_index": int(segment_idx),
                    }
                )
                seg_report = {
                    "scene": scene,
                    "video": video,
                    "pedestrian_ID": ped_id,
                    "raw_track_id": int(track_id),
                    "segment_index": int(segment_idx),
                    "first_frame": int(local_frames[0]),
                    "last_frame": int(local_frames[-1]),
                    "start_pos": points[0].tolist(),
                    "end_pos": points[-1].tolist(),
                }
                video_segments.append(seg_report)
                all_segments.append(seg_report)

        obsmat = pd.DataFrame(out_rows)
        if obsmat.empty:
            report["videos"][f"{scene}/{video}"] = {
                "status": "excluded",
                "reason": "no_valid_pedestrian_segments_after_downsampling",
            }
            continue
        obsmat = obsmat.sort_values(["frame_number", "pedestrian_ID"]).reset_index(drop=True)
        goal_table = make_goal_table(goal_rows)
        min_frame = int(obsmat["frame_number"].min())
        max_frame = int(obsmat["frame_number"].max())
        gt_list = frame_dicts_from_obsmat(obsmat, min_frame, max_frame)
        transformer_samples = build_transformer_samples(obsmat, goal_table)

        video_dir = processed_root / scene / video
        video_dir.mkdir(parents=True, exist_ok=True)
        obsmat[["frame_number", "pedestrian_ID", "pos_x", "pos_z", "pos_y", "v_x", "v_z", "v_y"]].to_csv(
            video_dir / "obsmat.txt", sep="\t", header=False, index=False
        )
        obsmat.to_pickle(video_dir / "obsmat_with_metadata.pkl")
        goal_table.to_pickle(video_dir / "Ped_Table.pkl")
        with (video_dir / "SDD_for_Metrics.pkl").open("wb") as handle:
            pickle.dump(gt_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with (video_dir / "environment_polygons.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "scene": scene,
                    "video": video,
                    "img_id": img_id,
                    "scale_m_per_px": scale,
                    "polygons_m": env_m,
                    "polygons_by_class_m": env_by_class_m,
                    "physical_obstacle_classes": PHYSICAL_OBSTACLE_CLASSES,
                    "polygon_class_counts": polygon_class_counts,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print(
            f"  wrote timelines: rows={len(obsmat)} segments={len(video_segments)} transformer_samples={len(transformer_samples)}",
            flush=True,
        )

        if args.skip_lidar:
            lidar_samples = [[], []]
        else:
            lidar_samples = build_lidar_samples(transformer_samples, env_m)
            print(f"  built lidar_samples={len(lidar_samples[0])}", flush=True)

        all_samples_by_video[f"{scene}/{video}"] = transformer_samples
        all_lidar_by_video[f"{scene}/{video}"] = lidar_samples
        all_transformer.extend(transformer_samples)
        all_lidar[0].extend(lidar_samples[0])
        all_lidar[1].extend(lidar_samples[1])

        global_obsmat = obsmat.copy()
        global_goal_table = goal_table.copy()
        global_frame_offset = next_global_frame - int(global_obsmat["frame_number"].min())
        global_obsmat["frame_number"] += global_frame_offset
        global_goal_table["First_Frame"] += global_frame_offset
        global_goal_table["Last_Frame"] += global_frame_offset
        next_global_frame = int(global_obsmat["frame_number"].max()) + 2
        all_obsmat.append(global_obsmat)
        all_goal_tables.append(global_goal_table)
        while len(all_gt) < global_frame_offset:
            all_gt.append({})
        all_gt.extend(gt_list)

        report["videos"][f"{scene}/{video}"] = {
            "status": "processed",
            "constrained_img_id": img_id,
            "constrained_image_shape_hw": [shape_h, shape_w],
            "polygon_resize_to_raw_factor": polygon_resize_to_raw,
            "scale_m_per_px": scale,
            "scale_certainty": scale_info["certainty"],
            "raw_pedestrian_rows": int(len(raw_ped)),
            "lost_pedestrian_rows": int((raw_ped["lost"] == 1).sum()),
            "valid_downsampled_rows": int(len(valid)),
            "segments": int(len(video_segments)),
            "split_tracks": int(local_split_count),
            "short_segments_dropped": int(local_short_segments),
            "frame_range_2p5fps": [min_frame, max_frame],
            "transformer_samples": int(len(transformer_samples)),
            "lidar_samples": int(len(lidar_samples[0])),
            "polygon_class_counts": polygon_class_counts,
        }

    if all_obsmat:
        root_obsmat = pd.concat(all_obsmat, ignore_index=True).sort_values(
            ["frame_number", "pedestrian_ID"]
        )
        root_obsmat[["frame_number", "pedestrian_ID", "pos_x", "pos_z", "pos_y", "v_x", "v_z", "v_y"]].to_csv(
            dataset_root / "obsmat.txt", sep="\t", header=False, index=False
        )
        root_obsmat.to_pickle(dataset_root / "obsmat_with_metadata.pkl")
    if all_goal_tables:
        pd.concat(all_goal_tables, ignore_index=True).to_pickle(dataset_root / "Ped_Table.pkl")
    with (dataset_root / "SDD_for_Metrics.pkl").open("wb") as handle:
        pickle.dump(all_gt, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with (transformer_root / "whole_sdd.pkl").open("wb") as handle:
        pickle.dump(all_transformer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (lidar_root / "whole_sdd.pkl").open("wb") as handle:
        pickle.dump(all_lidar, handle, protocol=pickle.HIGHEST_PROTOCOL)

    video_keys = sorted(all_samples_by_video)
    split_idx = max(1, int(len(video_keys) * 0.7))
    part1_keys = set(video_keys[:split_idx])
    part1_transformer = []
    part2_transformer = []
    part1_lidar = [[], []]
    part2_lidar = [[], []]
    for key in video_keys:
        if key in part1_keys:
            part1_transformer.extend(all_samples_by_video[key])
            part1_lidar[0].extend(all_lidar_by_video[key][0])
            part1_lidar[1].extend(all_lidar_by_video[key][1])
        else:
            part2_transformer.extend(all_samples_by_video[key])
            part2_lidar[0].extend(all_lidar_by_video[key][0])
            part2_lidar[1].extend(all_lidar_by_video[key][1])

    with (transformer_root / "SDD_train_part_1.pkl").open("wb") as handle:
        pickle.dump(part1_transformer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (transformer_root / "SDD_train_part_2.pkl").open("wb") as handle:
        pickle.dump(part2_transformer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (lidar_root / "SDD_train_part_1.pkl").open("wb") as handle:
        pickle.dump(part1_lidar, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (lidar_root / "SDD_train_part_2.pkl").open("wb") as handle:
        pickle.dump(part2_lidar, handle, protocol=pickle.HIGHEST_PROTOCOL)

    report["summary"] = {
        "processed_videos": len(video_keys),
        "total_transformer_samples": len(all_transformer),
        "total_lidar_samples": len(all_lidar[0]),
        "part1_videos": sorted(part1_keys),
        "part2_videos": [key for key in video_keys if key not in part1_keys],
        "split_track_events": len(split_events),
        "high_speed_jump_events": len(jump_events),
        "root_global_frame_range": [
            int(root_obsmat["frame_number"].min()) if all_obsmat else -1,
            int(root_obsmat["frame_number"].max()) if all_obsmat else -1,
        ],
        "root_metric_frames": len(all_gt),
        "root_ped_table_rows": int(sum(len(table) for table in all_goal_tables)),
    }
    report["split_track_events"] = split_events
    report["high_speed_jump_events"] = jump_events
    report["candidate_id_switches"] = candidate_id_switches(all_segments)

    with (dataset_root / "sdd_processing_report.json").open("w") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
