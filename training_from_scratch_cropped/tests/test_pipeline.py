"""Geometry, cache, cohort, contact estimator and exact training resume checks."""
import io
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from finetuning_pair.tests.test_pair_pipeline import make_scene
from training_from_scratch.tests.test_training import tiny_myriad
from training_from_scratch.fp32_dual_gpu.model import CachedEncoderSpec
from training_from_scratch.fp32_dual_gpu.optimizer import DeviceAdamW
from training_from_scratch.train import make_scheduler
from training_from_scratch.dataset import farthest_point_indices
from training_from_scratch_cropped.dataset import ArticulatedDataset, collate_articulated
from training_from_scratch_cropped.geometry import mask_crop, SquareCrop
from training_from_scratch_cropped.features import FeatureCache, fill_cache
from training_from_scratch_cropped.model import CachedFP32Model, training_loss
from training_from_scratch_cropped.traversal import PairCursor
from training_from_scratch_cropped.scoring import BestState, axis_error, checkpoint_score
from training_from_scratch_cropped.validation import trajectory_diagnostics
from training_from_scratch_cropped import fitting
from training_from_scratch_cropped.checkpoint_io import save_training_checkpoint, resume_checkpoint, inspect_checkpoint
from training_from_scratch_cropped.config import compare_config, parse_args
from training_from_scratch_cropped.selection import (
    joint_motion, select_validation, choose_representative, distribution_report, validate_cohort,
)

ROOT=Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT)
        self.root=Path(self.temp.name)
        make_scene(self.root,"train")
        make_scene(self.root,"val")

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, training=False):
        return ArticulatedDataset(self.root, split="train" if training else "val",training=training,
            image_size=32, boundary_margin=0,cache_dir=self.root/"scenes_cache")

    def cache(self, require=False):
        return FeatureCache(self.root/"features",dict(image_size=32,feature_cache_dtype="fp16",test=True),
                            require_cached=require,channels=24)

    def test_scene_order_pair_completeness_and_mid_epoch_resume(self):
        examples=[(s,(j,j+1,j)) for s,n in enumerate([1,4,2,3]) for j in range(n)]
        c=PairCursor(examples,42)
        seen=c.next_indices(3)
        saved=c.state_dict()
        r=PairCursor.from_state(saved,examples,42)
        while c.offset<len(examples):
            a,b=c.next_indices(3),r.next_indices(3)
            self.assertEqual(a,b);seen.extend(a)
        self.assertEqual(sorted(seen),list(range(len(examples))))
        scenes=[examples[i][0] for i in seen]
        self.assertEqual(len([s for i,s in enumerate(scenes) if i==0 or scenes[i-1]!=s]),4)
        first=seen
        c.next_indices(1)
        self.assertNotEqual(first,c.order())
        with self.assertRaises(ValueError):PairCursor.from_state(saved,examples[::-1],42)

    def test_fps_unique_padding_and_training_order(self):
        d=self.dataset(True)
        self.assertEqual(len(d),3)
        torch.manual_seed(17); a=[d[0] for _ in range(6)]
        torch.manual_seed(17); b=[d[0] for _ in range(6)]
        for x,y in zip(a,b):torch.testing.assert_close(x["reference_pixels"],y["reference_pixels"])
        self.assertGreater(len({s["reference_pixels"].numpy().tobytes() for s in a}),1)
        s=d[1];v=s["point_valid"]
        self.assertEqual(int(v.sum()),3)
        self.assertEqual(len(torch.unique(s["reference_pixels"][v],dim=0)),3)
        self.assertFalse(s["flow_loss_mask"].reshape(4,8)[:,~v].any())
        pts=np.array([[0,0],[1,0],[4,2],[2,5],[7,7]])
        ids=farthest_point_indices(pts,8,np.random.default_rng(4))
        self.assertEqual(len(set(ids)),5)
        for i in range(1,len(ids)):
            distance=((pts[:,None]-pts[ids[:i]])**2).sum(-1).min(-1)
            distance[ids[:i]]=-1
            self.assertEqual(ids[i],distance.argmax())

    def test_crop_roundtrip_reprojection_alignment_and_initial_only(self):
        d=self.dataset();s=d[0]
        c=SquareCrop(*s["crop_bounds"].tolist())
        px=s["reference_pixels"].numpy()
        np.testing.assert_allclose(c.unnormalize(c.normalize(px)),px,atol=1e-12)
        camera=s["relative_camera_points"].numpy()
        uv=camera@s["camera_intrinsic_model"].numpy().T
        uv=uv[...,:2]/uv[...,2:]
        valid=s["point_valid"].numpy()
        np.testing.assert_allclose(uv[:,valid]/31,s["trajectories"].numpy()[:,valid],atol=1e-6)
        torch.testing.assert_close(s["x"],d.initial_image(0)["x"])
        np.testing.assert_allclose(c.unnormalize(s["trajectories"].numpy()[:,valid]),s["source_trajectories"].numpy()[:,valid],atol=1e-5)
        p=d.files[0]
        with np.load(p) as ar:a={k:ar[k] for k in ar.files}
        a["link_transforms_world"][1:,:,0,3]+=20
        np.savez_compressed(p,**a)
        t=self.dataset()[0]
        torch.testing.assert_close(s["x"],t["x"])
        torch.testing.assert_close(s["crop_bounds"],t["crop_bounds"])
        parent=np.zeros((10,40),bool);child=parent.copy();parent[0,0]=1;child[-1,-1]=1
        c=mask_crop(parent,child,image_size=32)
        self.assertTrue(((c.normalize(np.array([[0,0],[39,9]]))>=-.02)&(c.normalize(np.array([[0,0],[39,9]]))<=1.02)).all())

    def test_offcrop_targets_retained_and_inverse_metrics(self):
        d=self.dataset();s=d[0]
        # Explicit off-crop but original-frame-valid motion.
        c=SquareCrop(10,10,8,32)
        truth=np.array([[[12.,12.]],[[22.,12.]],[[25.,12.]]])
        valid=np.ones((3,1),bool)
        cropped=c.normalize(truth)
        incrop=((cropped>=0)&(cropped<=1)).all(-1)
        metrics=trajectory_diagnostics(c.unnormalize(cropped+.01),truth,valid,incrop,(32,32))
        self.assertEqual(metrics["future_outside_crop"],2)
        self.assertEqual(metrics["original_valid_future"],2)
        self.assertGreater(metrics["pixel_ADE"],0)

    def test_cache_persistence_validation_coverage_and_fps_independence(self):
        cache=self.cache();d=self.dataset(True)
        image=d.initial_image(0)["x"].unsqueeze(0)
        self.assertIsNone(cache.read(image))
        row=dict(feat=torch.randn(1,24,2,2),pos=torch.randn(1,2,2,2),L=4)
        cache.write(image,row);path=cache.path(image);mtime=path.stat().st_mtime_ns
        cache.write(image,row);self.assertEqual(mtime,path.stat().st_mtime_ns)
        other=self.cache(True)
        self.assertEqual(other.read(image)["feat"].dtype,torch.float16)
        self.assertEqual(other.coverage([path])["remaining"],0)
        for _ in range(3):self.assertEqual(cache.path(d[0]["x"]),path)
        with patch("training_from_scratch_cropped.features.build_myriad",side_effect=AssertionError("DINO must stay unloaded")):
            paths,coverage=fill_cache(other,[(d,[0])],checkpoint=None,device="cpu",emit=lambda r:None)
        self.assertEqual(coverage["remaining"],0)
        torch.save({"namespace":cache.namespace,"embedding":dict(row,feat=torch.full((1,24,2,2),float('nan')))},path)
        with self.assertRaises(FileNotFoundError):other.read(image)
        self.assertIsNone(cache.read(image));self.assertEqual(cache.invalid,1)

    def test_axis_only_fixed_cohort_failures_ties_and_sign(self):
        self.assertEqual(axis_error([-1,0,0],[1,0,0]),0)
        self.assertEqual(axis_error([0,0,0],[1,0,0]),90)
        rows=[dict(index=0,forced_revolute_axis_world=[1,0,0],gt_axis_world=[1,0,0],predicted_type="fixed"),
              dict(index=1,gt_axis_world=[1,0,0])]
        self.assertEqual(checkpoint_score(rows,[0,1]),45)
        best=BestState();self.assertTrue(best.consider(45,1));self.assertFalse(best.consider(45,2))
        self.assertFalse(best.consider(46,3));self.assertTrue(best.consider(44,4))
        self.assertEqual(BestState(best.state_dict()).score,44)
        with self.assertRaises(ValueError):checkpoint_score(rows[:1],[0,1])
        with self.assertRaises(ValueError):compare_config({"crop_padding":.1},{"crop_padding":.2})

    def test_movable_joint_positions_exclude_fixed_edges(self):
        result=joint_motion([2,7,12],[2,12],np.array([[.3,.8],[0.,0.]]))
        self.assertEqual(result,{2:.3,7:0.,12:.8})

    def test_validation_manifest_exact_100_unique_stratified_and_reused(self):
        path=self.root/"val"/"scenes"/"example.npz"
        with np.load(path) as a:arrays={k:a[k] for k in a.files}
        arrays.update(movable_joint_indices=np.array([9,20]),joint_positions=np.array([[.3,.1]]*4+[[0.,0.]]),
                      joint_axes_world=np.broadcast_to([0.,0.,1.],(5,3,3)).copy())
        # Make both non-revolute children scoreable too: exclusion must depend
        # on joint type rather than their original lack of visible points.
        arrays["link_segmentation"][:,4:12,16:20] = 30
        arrays["link_segmentation"][:,13:20,22:29] = 99
        for i in range(102):
            meta=json.loads(str(arrays["metadata_json"].item()))
            meta["object"]["object_id"]=f"object_{i%12}"
            meta["object"]["category"]=f"category_{i%2}"
            meta["camera"]["azimuth_deg"]=i%8*45
            arrays["metadata_json"]=np.array(json.dumps(meta))
            np.savez_compressed(path if i==0 else path.with_name(f"scene_{i:03d}.npz"),**arrays)
        d=self.dataset()
        manifest=select_validation(d,self.root/"validation_manifest.json",seed=42)
        self.assertEqual(len(manifest["pairs"]),100)
        self.assertEqual(len({r["index"] for r in manifest["pairs"]}),100)
        self.assertEqual(manifest["scene_count"],100)
        self.assertEqual(manifest["object_count"],12)
        self.assertEqual(len(manifest["axis_cohort"]),100)
        self.assertEqual({r["joint_type"] for r in manifest["pairs"]},{"revolute"})
        self.assertEqual(manifest["excluded"]["joint_type_prismatic"],102)
        self.assertEqual(manifest["excluded"]["joint_type_fixed"],102)
        self.assertLess(manifest["distribution"]["object_id"]["total_variation_distance"],.02)
        self.assertEqual(select_validation(d,self.root/"validation_manifest.json",seed=42),manifest)
        with self.assertRaises(ValueError):select_validation(d,self.root/"validation_manifest.json",seed=43)
        invalid=dict(manifest,pairs=[dict(r) for r in manifest["pairs"]])
        invalid["pairs"][0]["joint_type"]="fixed"
        with self.assertRaises(ValueError):validate_cohort(invalid)
        from training_from_scratch_cropped.validation import evaluate
        with self.assertRaises(ValueError):evaluate(None,d,invalid)

    def test_proportional_selection_preserves_imbalanced_data_distribution(self):
        rows=[]
        for i in range(1000):
            obj="common" if i<800 else "less_common" if i<950 else "rare"
            rows.append(dict(index=i,scene=i,object_id=obj,category="arm",joint_type="revolute",
                joint_ordinal=i%4+1,chain_depth=i%4+1,viewpoint=str(i%8),
                motion_bin=i%4,size_bin=(i//4)%4,magnification_bin=(i//16)%4,
                motion=float(i),part_size=float(i%100),magnification=float(i%20)))
        selected,quotas=choose_representative(rows,count=100,seed=42)
        self.assertEqual(quotas,{"common":80,"less_common":15,"rare":5})
        from collections import Counter
        self.assertEqual(dict(Counter(r["object_id"] for r in selected)),quotas)
        self.assertEqual(len({r["scene"] for r in selected}),100)
        self.assertEqual(selected,choose_representative(rows,count=100,seed=42)[0])
        report=distribution_report(rows,selected)
        self.assertEqual(report["object_id"]["total_variation_distance"],0.)
        for field in ("motion_bin","size_bin","magnification_bin","viewpoint"):
            self.assertLessEqual(report[field]["total_variation_distance"],.05)

    def test_estimator_boundary_and_contact_fallback(self):
        self.assertNotIn("gt",inspect.signature(fitting.fit_predictions).parameters)
        xyz=np.zeros((8,3));uv=np.zeros((5,8,2))
        result=dict(fit=dict(diagnostics={},fits={"revolute":None}))
        with patch.object(fitting,"estimate_motion",return_value=result) as spy:
            out=fitting.fit_predictions(xyz,uv,np.eye(3),np.eye(4),initial_depth=np.ones((8,8)),
                 initial_segmentation=np.zeros((8,8)),pair_labels=[0,1])
            self.assertIn("trajectory_only",out["diagnostics"]["contact_fallback"])
            self.assertEqual(set(spy.call_args.kwargs),{"initial_depth","initial_segmentation","pair_labels"})
            self.assertTrue(spy.call_args.args[2].all())

    def test_contact_prior_is_soft_axis_line_and_handles_occlusion(self):
        from testing_joint_fitting.geometric import contact_evidence, refine_revolute
        from testing_joint_fitting.test_conditioning import ConditioningTests
        xyz,pivot,k,e,uv,valid=ConditioningTests().scene()
        # An uncertain visible contact away from the actual pivot must not
        # impose that point as the pivot. The residual is distance to an axis
        # line, so shifting the stored contact center along that line is a gauge.
        seed=dict(params=np.r_[0.,0.,1.,pivot,.2,.35,.5])
        contact=dict(center=pivot+np.array([.1,.1,1.]),whitening=np.eye(3))
        fitted=refine_revolute(seed,xyz,uv[2:],valid[2:],k,e,contact=contact,contact_start=True)
        self.assertLess(axis_error(fitted["params"][:3],[0,0,1]),.1)
        self.assertLess(fitted["rmse_px"],.01)
        self.assertGreater(np.linalg.norm(fitted["params"][3:6]-contact["center"]),.1)
        hidden=dict(depth=np.ones((8,8)),segmentation=np.ones((8,8)),pair_labels=[0,1])
        self.assertIsNone(contact_evidence(hidden,k,e))

    def test_latest_exact_resume_rng_optimizer_scheduler_cursor(self):
        d=self.dataset(True);cache=self.cache()
        encoder=tiny_myriad().eval()
        for i in range(len(d)):
            image=d.initial_image(i)["x"][None]
            with torch.no_grad():cache.write(image,encoder.embed_image(image))
        ref=dict(identity="in_memory_test_encoder")
        def setup():
            bare=tiny_myriad();bare.image_embedder=CachedEncoderSpec(image_size=32,embed_dim=24)
            model=CachedFP32Model(bare).place(torch.device("cpu"));model.frozen_encoder_reference=ref
            model.configure_cropped_cache(self.cache(True))
            opt=DeviceAdamW(model.trainable_parameters(),device="cpu",lr=1e-3,betas=(.9,.99))
            return model,opt,make_scheduler(opt,10,0)
        model,opt,sched=setup();cursor=PairCursor(d.examples,4);best=BestState();best.consider(30,1)
        def update(m,o,sc,c):
            indices=c.next_indices(1);batch=collate_articulated([d[i] for i in indices])
            o.zero_grad(set_to_none=True);loss=training_loss(m,batch);loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
            o.capture_model_gradients();o.step();sc.step();return float(loss)
        update(model,opt,sched,cursor)
        from finetuning_pair.geometry import TARGET_DEFINITION
        config=dict(target_definition=TARGET_DEFINITION,freeze_dino=True,dino_identity=ref["identity"],precision="fp32")
        latest=self.root/"latest.pt"
        # Exercise actual Torch checkpoint serialization and exact continuation
        # without creating any checkpoint files, even temporary ones.
        serialized={}
        from training_from_scratch_cropped.checkpoint_io import _read_checkpoint as read_serialized_checkpoint
        def save_memory(payload,path):
            stream=io.BytesIO()
            torch.save(payload,stream)
            serialized[str(path)]=stream.getvalue()
        def read_memory(path):
            return read_serialized_checkpoint(io.BytesIO(serialized[str(path)]))
        with patch("training_from_scratch_cropped.checkpoint_io._atomic_save",side_effect=save_memory), \
             patch("training_from_scratch_cropped.checkpoint_io._read_checkpoint",side_effect=read_memory), \
             patch("training_from_scratch_cropped.checkpoint_io.resolve_encoder_reference",return_value=ref):
            save_training_checkpoint(latest,model,opt,sched,step=1,train_config=config,
                data_state=cursor.state_dict(),metrics=dict(best_state=best.state_dict()))
            baseline=update(model,opt,sched,cursor)
            states={n:p.detach().clone() for n,p in model.named_parameters()}
            resumed,o2,s2=setup();self.assertEqual(resume_checkpoint(latest,resumed,o2,s2),1)
            restored=PairCursor.from_state(inspect_checkpoint(latest)["data_state"],d.examples,4)
            self.assertEqual(update(resumed,o2,s2,restored),baseline)
            for n,p in resumed.named_parameters():torch.testing.assert_close(p,states[n],rtol=0,atol=0)
            self.assertEqual(sched.state_dict(),s2.state_dict())
            self.assertEqual(restored.state_dict(),cursor.state_dict())
            self.assertEqual(BestState(inspect_checkpoint(latest)["metrics"]["best_state"]).score,30)
        self.assertFalse(latest.exists())

    def test_smoke_policy_and_temporary_output_cleanup(self):
        from training_from_scratch_cropped import train
        import contextlib
        args,_=parse_args(["--run-steps","2"])
        self.assertFalse(args.save_checkpoints)
        args,_=parse_args([])
        self.assertTrue(args.save_checkpoints)
        output=[]
        def fake_run(args,saved):
            self.assertFalse(args.save_checkpoints)
            self.assertEqual(args.run_steps,2)
            self.assertEqual(args.max_train_scenes,1)
            self.assertTrue(args.feature_cache.is_relative_to(args.output_dir))
            output.append(args.output_dir)
            (args.output_dir/"diagnostic.json").write_text("{}")
            return {"checked":True}
        with patch.object(train,"run",side_effect=fake_run):
            self.assertEqual(train.main(["--smoke-test"]),{"checked":True})
        self.assertTrue(all(not path.exists() for path in output))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--smoke-test","--save-checkpoints"])



if __name__ == "__main__":unittest.main()
