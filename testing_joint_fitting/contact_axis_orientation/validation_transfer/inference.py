"""Reuse existing inference unchanged, restricted to the user's 50-scene cohort."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import training_from_scratch.full_evaluation.run as core
from finetuning_pair.evaluate import write_json

ROOT = Path(__file__).resolve().parent
ORIGINAL_DATASET = core.make_dataset


def selected_dataset(args, cfg):
    data = ORIGINAL_DATASET(args, cfg)
    selection = json.loads((ROOT/"subset50.json").read_text())
    selected = set(selection["paths"])
    indices = [i for i, path in enumerate(data.files) if str(path.resolve()) in selected]
    assert len(indices) == 50 and len(selected) == 50
    data.files = [data.files[i] for i in indices]
    data.pairs = [data.pairs[i] for i in indices]
    data.examples = [(scene, pair) for scene, pairs in enumerate(data.pairs) for pair in pairs]
    data.training_indices = [i for i, pairs in enumerate(data.pairs) if pairs]
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--checkpoint", type=Path, default=Path("training_from_scratch/runs/test1_best_full_20261017/checkpoint.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("training_from_scratch/runs/val50_contact_axis_transfer_20261021"))
    parser.add_argument("--feature-cache", type=Path, default=Path("training_from_scratch/feature_cache"))
    parser.add_argument("--cache-dir", type=Path, default=Path("training_from_scratch/cache"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.data_root = None; args.split = "val"; args.seed = 42; args.ensemble_size = 5
    args.diffusion_steps = 16; args.modes = ["without_poke"]; args.min_path_pixels = 1.; args.threads = 1
    args.output_dir, args.feature_cache, args.cache_dir = map(core.artifact_path,
        (args.output_dir, args.feature_cache, args.cache_dir))
    # Process-local adapter only: no production file or algorithm is modified.
    core.make_dataset = selected_dataset
    if args.worker_index is not None: return core.worker(args)
    manifest = core.prepare(args)
    assert len(manifest["scenes"]) == 50
    write_json(args.output_dir/"subset50.json", json.loads((ROOT/"subset50.json").read_text()))
    if args.prepare_only: return
    processes, handles = [], []
    try:
        for index in range(len(args.devices)):
            handle = (args.output_dir/f"worker_{index}.log").open("a")
            handles.append(handle)
            processes.append(subprocess.Popen([sys.executable, "-B", "-u", "-m", __name__.replace("__main__",
                "testing_joint_fitting.contact_axis_orientation.validation_transfer.inference"),
                *sys.argv[1:], "--worker-index", str(index)], stdout=handle, stderr=subprocess.STDOUT))
        while any(process.poll() is None for process in processes):
            time.sleep(20)
            statuses = [json.loads(path.read_text()) for path in args.output_dir.glob("status_worker_*.json")]
            print(json.dumps(dict(event="inference_progress", workers=statuses)), flush=True)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError("Inference failed; see worker logs")
    finally:
        for process in processes:
            if process.poll() is None: process.terminate()
        for handle in handles: handle.close()
    # The completion helper also needs the same process-local dataset adapter.
    import training_from_scratch.full_evaluation.complete_unscorable as completion
    completion.make_dataset = selected_dataset
    completion.complete(args.output_dir, args.devices[-1])
    from training_from_scratch.full_evaluation.audit import audit
    audit(args.output_dir)


if __name__ == "__main__": main()
