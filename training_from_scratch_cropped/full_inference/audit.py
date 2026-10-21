"""Check archive completeness, source hashes, coordinates and metric calculations."""
import argparse
import json
from pathlib import Path
import numpy as np
from finetuning_pair.evaluate import file_hash
from training_from_scratch_cropped.config import write_json
from training_from_scratch_cropped.geometry import SquareCrop
from .common import DEFAULT_OUT


def audit(output):
    output = Path(output); manifest = json.loads((output/"run.json").read_text())
    assert file_hash(output/"checkpoint.pt")==manifest["checkpoint_sha256"]
    checked = {}; max_crop_roundtrip = 0.; max_metric_difference = 0.; max_anchor_projection = 0.
    observed = set()
    for split, info in manifest["splits"].items():
        assert len(info["scenes"]) == (320 if split=="test1" else 80)
        for descriptor in info["scenes"]:
            path = output/split/descriptor["archive"]
            assert file_hash(path)==descriptor["sha256"], path
            with np.load(path, allow_pickle=False) as source:
                for key in ("rgb", "depth", "depth_valid_mask", "link_segmentation", "mesh_points", "mesh_normals",
                            "mesh_point_link_labels", "joint_axes_world", "joint_origins_world", "link_transforms_world"):
                    assert key in source.files
        arrays_count = 0; unscoreable = 0
        for index in range(len(info["pairs"])):
            row = json.loads((output/split/"records"/f"pair_{index:05d}.json").read_text())
            assert row["pair_index"]==index and row["seed"]==manifest["seed"]+index
            observed.add((split, row["scene_index"]))
            assert row["status"]=="predicted", row
            with np.load(output/split/row["prediction_file"], allow_pickle=False) as arrays:
                pred = arrays["predictions_crop_normalized"]
                source_pred = arrays["predictions_source_pixels"]
                truth = arrays["ground_truth_source_pixels"]
                valid = arrays["valid"]
                crop = SquareCrop(*arrays["crop_bounds"].tolist())
                assert pred.shape==(manifest["ensemble_size"], 5, valid.shape[1], 2)
                assert np.isfinite(pred).all() and np.isfinite(source_pred).all() and np.isfinite(truth[valid]).all()
                np.testing.assert_allclose(crop.unnormalize(pred), source_pred, rtol=0, atol=1e-10)
                error = float(np.max(np.abs(crop.normalize(source_pred)-pred)))
                max_crop_roundtrip = max(max_crop_roundtrip, error)
                # The official crop inverse starts with float32 model outputs.
                # Undoing it in float64 retains the original float32 rounding;
                # bound that error by precision, not an exact-float64 tolerance.
                tolerance = 8*np.finfo(pred.dtype).eps*max(1., float(np.max(np.abs(pred))))
                assert error<tolerance, (row["prediction_file"], error, tolerance)
                np.testing.assert_allclose(arrays["crop_transform"]@arrays["crop_inverse"], np.eye(3), atol=1e-12)
                np.testing.assert_allclose(arrays["camera_intrinsic_model"], arrays["crop_transform"]@arrays["camera_intrinsic"], atol=1e-10)
                np.testing.assert_allclose(pred[:, 0], np.broadcast_to(arrays["ground_truth_crop_normalized"][0], pred[:, 0].shape), atol=1e-6, rtol=0)
                camera = arrays["reference_world"]@arrays["camera_extrinsic"][:3, :3].T+arrays["camera_extrinsic"][:3, 3]
                projected = camera@arrays["camera_intrinsic"].T
                projected = projected[:, :2]/projected[:, 2:]
                max_anchor_projection = max(max_anchor_projection, float(np.max(np.abs(projected-arrays["reference_source_pixels"])) ))
                for key in ("input_image", "cropped_rgb", "parent_mask", "child_mask", "relative_camera_points",
                    "ground_truth_actual_source_pixels", "local_points", "point_indices", "point_link_labels"):
                    assert key in arrays.files
                mask = valid[1:]
                assert row["scoreable"] == bool(mask.any())
                if mask.any():
                    for slot, metrics in enumerate(row["trajectory_metrics"]["per_hypothesis"]):
                        distances = np.linalg.norm(source_pred[slot]-truth, axis=-1)
                        ade = float(distances[1:][mask].mean())
                        max_metric_difference = max(max_metric_difference, abs(ade-metrics["pixel_ADE"]))
                        np.testing.assert_allclose(ade, metrics["pixel_ADE"], rtol=1e-12, atol=1e-10)
                        rmse = float(np.sqrt(np.mean(distances[1:][mask]**2)))
                        np.testing.assert_allclose(rmse, metrics["source_RMSE_pixels"], rtol=1e-12, atol=1e-10)
                        if valid[-1].any():
                            np.testing.assert_allclose(distances[-1][valid[-1]].mean(), metrics["source_FDE_pixels"], rtol=1e-12, atol=1e-10)
                        else:
                            assert metrics["source_FDE_pixels"] is None
                        path_length = np.linalg.norm(np.diff(truth.astype(float), axis=0), axis=-1).sum(0)
                        moving = valid.all(0) & (path_length>=1.)
                        if row["gt_type"]!="fixed" and moving.any():
                            percent = np.mean(100*distances[1:,moving].mean(0)/path_length[moving])
                            np.testing.assert_allclose(percent, metrics["GT_path_normalized_error_percent"], rtol=1e-12, atol=1e-10)
                        else:
                            assert metrics["GT_path_normalized_error_percent"] is None
                else: unscoreable += 1
            arrays_count += 1
        assert len([path for path in (output/split/"records").glob("*.json")])==len(info["pairs"])
        assert len(list((output/split/"predictions").glob("*.npz")))==arrays_count
        checked[split] = dict(scenes=len(info["scenes"]), pairs=arrays_count, unscoreable_predictions_saved=unscoreable,
            excluded_source_joints=sum(not item["eligible"] for item in info["inventory"]))
    catalog = json.loads((output/"visualization_catalog.json").read_text())["examples"]
    assert len(catalog)==sum(entry["pairs"] for entry in checked.values())
    assert len({(row["split"], row["pair_index"]) for row in catalog})==len(catalog)
    # Source camera rotations are stored in float32 and are not perfectly
    # orthogonal; the existing backproject helper uses their transpose inverse.
    assert max_anchor_projection<1e-3
    result = dict(status="passed", checkpoint_sha256_verified=True, source_archives_sha256_verified=True,
        splits=checked, scenes_with_predictions=len(observed), visualization_catalog_entries=len(catalog),
        max_crop_normalized_roundtrip_error=max_crop_roundtrip, max_ADE_recalculation_difference=max_metric_difference,
        max_anchor_reprojection_error_pixels=max_anchor_projection, fitting_run=False)
    write_json(output/"audit.json", result)
    # Keep the actual reporting/verification tools alongside the original
    # inference-code snapshot, without changing that frozen snapshot.
    tools = {}
    for name in ("report.py", "audit.py", "preview.py"):
        source = Path(__file__).parent/name
        destination = output/"postprocessing_code"/name
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(source.read_text())
        tools[name] = file_hash(source)
    write_json(output/"postprocessing_code.json", tools)
    print(json.dumps(result), flush=True)
    return result


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUT)
    audit(parser.parse_args().output)
