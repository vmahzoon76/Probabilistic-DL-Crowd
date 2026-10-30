import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_sdd_dataset import build_lidar_samples, build_transformer_samples


def build_one_video(video_key: str, dataset_root: Path) -> dict:
    scene, video = video_key.split("/")
    video_dir = dataset_root / "processed" / scene / video
    out_path = video_dir / "lidar.pkl"
    if out_path.exists():
        with out_path.open("rb") as handle:
            x, y = pickle.load(handle)
        return {
            "video": video_key,
            "samples": int(x.shape[0]),
            "x_shape": list(x.shape),
            "y_shape": list(y.shape),
            "path": str(out_path),
            "skipped_existing": True,
        }

    obsmat = pd.read_pickle(video_dir / "obsmat_with_metadata.pkl")
    goals = pd.read_pickle(video_dir / "Ped_Table.pkl")
    with (video_dir / "environment_polygons.pkl").open("rb") as handle:
        env = pickle.load(handle)["polygons_m"]

    transformer_samples = build_transformer_samples(obsmat, goals)
    lidar_data = build_lidar_samples(transformer_samples, env)
    x = np.asarray(lidar_data[0], dtype=np.float32)
    y = np.asarray(lidar_data[1], dtype=np.float32)
    with out_path.open("wb") as handle:
        pickle.dump([x, y], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        "video": video_key,
        "samples": int(x.shape[0]),
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "path": str(out_path),
        "skipped_existing": False,
    }


def concat_video_lidar(video_keys: list[str], dataset_root: Path) -> list[np.ndarray]:
    xs = []
    ys = []
    for video_key in video_keys:
        scene, video = video_key.split("/")
        with (dataset_root / "processed" / scene / video / "lidar.pkl").open("rb") as handle:
            x, y = pickle.load(handle)
        xs.append(x)
        ys.append(y)
    if not xs:
        return [np.empty((0, 1444), dtype=np.float32), np.empty((0, 2), dtype=np.float32)]
    return [np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="Datasets/SDD")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    with (dataset_root / "sdd_processing_report.json").open() as handle:
        report = json.load(handle)

    video_keys = [
        key for key, value in report["videos"].items() if value.get("status") == "processed"
    ]
    lidar_report = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(build_one_video, video_key, dataset_root): video_key
            for video_key in video_keys
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            video_key = futures[future]
            info = future.result()
            status = "skipped" if info.get("skipped_existing") else "built"
            print(
                f"[{completed}/{len(video_keys)}] {status} lidar {video_key}: samples={info['samples']} x_shape={info['x_shape']}",
                flush=True,
            )
            lidar_report.append(info)

    lidar_root = dataset_root / "data_for_model" / "lidar"
    lidar_root.mkdir(parents=True, exist_ok=True)
    part1_keys = report["summary"]["part1_videos"]
    part2_keys = report["summary"]["part2_videos"]
    whole = concat_video_lidar(video_keys, dataset_root)
    part1 = concat_video_lidar(part1_keys, dataset_root)
    part2 = concat_video_lidar(part2_keys, dataset_root)

    for name, data in [
        ("whole_sdd.pkl", whole),
        ("SDD_train_part_1.pkl", part1),
        ("SDD_train_part_2.pkl", part2),
    ]:
        with (lidar_root / name).open("wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    report["summary"]["total_lidar_samples"] = int(whole[0].shape[0])
    report["lidar"] = {
        "storage": "pickle [X, Y] where X and Y are float32 numpy arrays",
        "whole_x_shape": list(whole[0].shape),
        "whole_y_shape": list(whole[1].shape),
        "part1_x_shape": list(part1[0].shape),
        "part2_x_shape": list(part2[0].shape),
        "videos": lidar_report,
    }
    with (dataset_root / "sdd_processing_report.json").open("w") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
