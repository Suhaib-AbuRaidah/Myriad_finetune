"""Regression tests for 3D cancellation, conditioning and model target routing."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

from finetuning_pair.conditioning import blend_roles, PARENT_COLOR, CHILD_COLOR, normalized_pixels
from finetuning_pair.dataset import PairDataset, collate_pairs
from finetuning_pair.geometry import relative_camera_points, project_camera_points, TARGET_DEFINITION
from finetuning_pair.inference import prepare_inputs, parse_args as inference_args
from finetuning_pair.train import training_loss, make_attention_mask, parse_args as train_args
from finetuning_pair.testing import sample_rollout, run_test_subset, TestConfig as RolloutConfig
from finetuning_pair.validation import evaluate_validation, ValidationConfig
from finetuning_pair.preprocess import verify_sample
from finetuning_pair.checkpoint_io import inspect_checkpoint

ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / '.test_artifacts'


def pose(angle=0, translation=(0, 0, 0)):
    c, s = np.cos(angle), np.sin(angle)
    out = np.eye(4)
    out[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    out[:3, 3] = translation
    return out


def make_scene(root, split='val', name='example.npz'):
    h, w = 24, 32
    yy, xx = np.indices((h, w))
    rgb = np.stack((xx*7, yy*9, (xx+yy)*4), axis=-1).astype(np.uint8)
    seg = np.full((h, w), -2, dtype=np.int16)
    seg[3:20, 5:9] = -1
    seg[3:20, 9:15] = 10
    seg[8, 16:19] = 30  # Only three unique child candidates, no duplicated padding.
    seg[12, 20] = 99
    e = pose(.3, (.2, -.3, .5))
    k = np.array([[30., 0, 15.5], [0, 32., 11.5], [0, 0, 1]])
    transforms = []
    for t in np.linspace(0, 1, 5):
        base = pose()
        parent = pose(.12*t, (.03*t, .02*t, .1*t)) @ pose(.2, (.1, -.2, .2))
        child = parent @ pose(translation=(.12*(1-t), .04, .1))
        fixed = child @ pose(.1, (.04, .01, 0))
        transforms.append([base, parent, child, fixed])
    metadata = dict(sample_id=name, object={'object_id': 'synthetic'},
                    camera={'extrinsic_convention': 'T_camera_world (world coordinates to camera coordinates)'},
                    state_semantics={'state_4': 'zero-joint rest state'})
    folder = root / split / 'scenes'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    np.savez_compressed(path, schema_version=np.array(1), metadata_json=np.array(json.dumps(metadata)),
        rgb=np.repeat(rgb[None], 5, axis=0), depth=np.full((5,h,w),3,dtype=np.float32),
        depth_valid_mask=np.ones((5,h,w),dtype=bool), link_segmentation=np.repeat(seg[None],5,axis=0),
        link_indices=np.array([-1,10,30,99]), link_transforms_world=np.array(transforms),
        camera_intrinsic=k, camera_extrinsic=e, state_fractions=np.linspace(0,1,5),
        joint_indices=np.array([9,20,44]), joint_parent_link_indices=np.array([-1,10,30]),
        joint_child_link_indices=np.array([10,30,99]), joint_types=np.array([0,1,4]))
    return path, rgb, seg


class SpyDistribution:
    def __init__(self, model): self.model = model
    def __getitem__(self, index): self.model.slice = index; return self
    def loss(self, target):
        self.model.target = target.detach().clone()
        return (target - self.model.weight).square().mean(-1)


class SpyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.01))
    def embed_image(self, image):
        self.image = image.clone()
        return {'L': 1, 'feat': image.mean((-1,-2))}
    def forward(self, **kwargs):
        self.inputs = kwargs
        return SpyDistribution(self)
    def predict_simulate(self, *, n_traj, ts, given_pos, **kwargs):
        known = given_pos.reshape(len(ts), -1, n_traj, 2)
        result = known[:, :1].expand(-1, ts.shape[1], -1, -1).clone()
        result[:, :known.shape[1]] = known
        self.given_pos = given_pos.clone()
        return result


class PairPipelineTests(unittest.TestCase):
    def setUp(self):
        ARTIFACT_ROOT.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ARTIFACT_ROOT)
        self.root = Path(self.temp.name)
        self.path, self.rgb, self.seg = make_scene(self.root)
    def tearDown(self): self.temp.cleanup()
    def ds(self, **kwargs): return PairDataset(self.root, split=kwargs.pop('split', 'val'), image_size=24, boundary_margin=0, **kwargs)

    def test_3d_parent_cancellation_with_nonidentity_camera(self):
        parent = np.array([pose(.4+t, (t, -2*t, 1+t)) for t in [0., .5, 1.]])
        child = np.array([p @ pose(translation=(.4*(1-t), .1, 2)) for p,t in zip(parent,[0.,.5,1.])])
        e = pose(-.7, (1, .2, 2))
        local = np.array([[.1,.2,.3],[-.2,.4,1.]])
        relative, actual = relative_camera_points(local, parent, child, e)
        expected = np.array([(e @ parent[0] @ pose(translation=(.4*(1-t), .1, 2)) @ np.c_[local,np.ones(2)].T).T for t in [0.,.5,1.]])
        np.testing.assert_allclose(relative, expected, atol=1e-12)
        np.testing.assert_allclose(relative[0], actual[0], atol=1e-12)
        self.assertGreater(np.linalg.norm(relative[-1]-actual[-1]), 1.)
        # A fixed child relative transform must be stationary despite upstream motion.
        fixed = parent @ pose(.8, (.2, .3, 2))
        frozen, _ = relative_camera_points(local, parent, fixed, e)
        np.testing.assert_allclose(frozen, np.repeat(frozen[:1], len(parent), axis=0), atol=1e-12)

    def test_roles_only_and_texture_blending(self):
        parent, child = self.seg==10, self.seg==30
        original = self.rgb.copy()
        out = blend_roles(self.rgb, parent, child)
        np.testing.assert_array_equal(self.rgb, original)
        np.testing.assert_array_equal(out[~(parent|child)], self.rgb[~(parent|child)])
        np.testing.assert_allclose(out[parent], .15*self.rgb[parent]+.85*PARENT_COLOR, atol=2e-5)
        np.testing.assert_allclose(out[child], .15*self.rgb[child]+.85*CHILD_COLOR, atol=2e-5)
        np.testing.assert_array_equal(blend_roles(self.rgb,parent,child,0), self.rgb)
        with self.assertRaises(ValueError): blend_roles(self.rgb,parent,parent)

    def test_enumeration_child_only_padding_and_relative_fixed_joint(self):
        ds = self.ds()
        self.assertEqual(len(ds), 3)
        self.assertEqual([ds[i]['pair_labels'].tolist() for i in range(3)], [[-1,10],[10,30],[30,99]])
        for i in range(3):
            sample = ds[i]
            self.assertEqual(sample['num_tracks'], 8)
            self.assertTrue((sample['point_link_labels'] == sample['pair_labels'][1]).all())
            self.assertTrue(torch.equal(sample['trajectories'], ds[i]['trajectories']))
            verify_sample(sample)
            self.assertFalse(sample['visibility_known'][1:].any())
        child = ds[1]
        self.assertEqual(child['point_valid'].sum().item(), 3)
        self.assertEqual(len(torch.unique(child['reference_pixels'][child['point_valid']], dim=0)), 3)
        self.assertFalse(child['flow_loss_mask'].reshape(4,8)[:,3:].any())
        fixed = ds[2]
        mask = fixed['point_valid']
        torch.testing.assert_close(fixed['trajectories'][:,mask], fixed['trajectories'][:1,mask].expand(5,-1,-1))
        self.assertGreater((fixed['actual_trajectories'][-1,mask]-fixed['actual_trajectories'][0,mask]).abs().max(),.001)

    def test_training_random_pair_and_repeatable_sampling_stream(self):
        make_scene(self.root, 'train')
        ds = self.ds(split='train')
        self.assertEqual(len(ds),1)
        torch.manual_seed(21)
        first = [ds[0] for _ in range(12)]
        self.assertEqual(len({tuple(s['pair_labels'].tolist()) for s in first}),3)
        torch.manual_seed(21)
        second = [ds[0] for _ in range(12)]
        for a,b in zip(first,second):
            torch.testing.assert_close(a['pair_labels'],b['pair_labels'])
            torch.testing.assert_close(a['trajectories'],b['trajectories'])

    def test_batch_teacher_forced_positions_flows_and_masked_loss(self):
        ds = self.ds()
        batch = collate_pairs([ds[0], ds[1]])
        truth = batch['trajectories']
        torch.testing.assert_close(batch['pos_poke'],truth[:,:-1].reshape(2,-1,2))
        torch.testing.assert_close(batch['pos_query'],batch['pos_poke'])
        expected_flow = (truth[:,1:]-truth[:,:-1]).reshape(2,-1,2)
        expected_flow[~batch['flow_loss_mask']] = 0
        torch.testing.assert_close(batch['flow_poke'],expected_flow)
        torch.testing.assert_close(batch['flow_query'],expected_flow)
        model = SpyModel()
        loss = training_loss(model,batch,amp_dtype='none')
        loss.backward()
        self.assertTrue(torch.isfinite(model.weight.grad))
        torch.testing.assert_close(model.target,expected_flow)
        length = expected_flow.shape[1]
        torch.testing.assert_close(model.inputs['pos'][:,:length],batch['pos_poke'])
        torch.testing.assert_close(model.inputs['flow'][:,:length],expected_flow)
        self.assertEqual(model.slice, (slice(None), slice(length,None)))
        invalid = ~batch['motion_token_mask'][1]
        mask = model.inputs['mask'][1,0]
        self.assertFalse(mask[:,1:1+length][:,invalid].any())
        self.assertFalse(mask[1:1+length][invalid].any())

    def test_targets_outside_image_remain_masked_and_can_be_skipped(self):
        with np.load(self.path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays['link_transforms_world'][1:,1,0,3] += 100
        np.savez_compressed(self.path,**arrays)
        sample = self.ds()[0]
        self.assertFalse(sample['flow_loss_mask'].any())
        self.assertTrue(sample['position_valid'][0].all())
        self.assertFalse(sample['position_valid'][1:].any())
        from finetuning_pair.testing import _scene_inputs
        self.assertIsNone(_scene_inputs(sample,False,False))

    def test_cache_separates_pairs_opacity_target_and_source(self):
        cache = self.root / 'cache'
        ds = self.ds(cache_dir=cache)
        a,b = ds[0], ds[1]
        self.assertNotEqual(a['cache_key'], b['cache_key'])
        with patch.object(ds, '_load', side_effect=AssertionError('should use cache')):
            torch.testing.assert_close(ds[0]['trajectories'],a['trajectories'])
        changed = self.ds(cache_dir=cache,overlay_alpha=.2)
        self.assertNotEqual(changed[0]['cache_key'],a['cache_key'])
        bad = dict(a,target_definition='old_actual_targets')
        torch.save(bad,cache / (a['cache_key']+'.pt'))
        with self.assertRaisesRegex(ValueError,'Incompatible relative cache'): ds[0]
        with self.assertRaisesRegex(ValueError,'inside'): self.ds(cache_dir='/tmp/not_pair_project')

    def test_rgb_masks_points_only_inference_equals_training(self):
        sample = self.ds()[1]
        mask = sample['point_valid']
        image, known = prepare_inputs(self.rgb,self.seg==10,self.seg==30,
                                      sample['reference_pixels'][mask].numpy(),image_size=24)
        torch.testing.assert_close(image,sample['x'])
        torch.testing.assert_close(known[0],sample['trajectories'][0,mask])
        image2,known2 = prepare_inputs(self.rgb,self.seg==10,self.seg==30,known[0].numpy(),
                                      image_size=24,point_space='normalized-model')
        torch.testing.assert_close(image,image2)
        torch.testing.assert_close(known,known2)
        with self.assertRaisesRegex(ValueError,'child mask'):
            prepare_inputs(self.rgb,self.seg==10,self.seg==30,np.array([[10,10]]),image_size=24)
        model = SpyModel()
        pred = sample_rollout(model,image,known,sample['state_times'],ensemble_size=2,diffusion_steps=1,seed=2,amp_dtype='none')
        self.assertEqual(pred.shape,(2,5,3,2))
        self.assertEqual(model.given_pos.shape,(2,3,2))

    def test_validation_testing_and_persisted_pair_metadata(self):
        ds, model = self.ds(include_final_image=True), SpyModel()
        before = torch.random.get_rng_state().clone()
        result = evaluate_validation(model, ds, ValidationConfig(num_scenes=3,ensemble_size=5,diffusion_steps=1,initial_poke=False,amp_dtype='none'))
        self.assertEqual(result['pairs_evaluated'],3)
        self.assertEqual(result['target_definition'],TARGET_DEFINITION)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertTrue(model.training)
        summary = run_test_subset(model,ds,RolloutConfig(num_scenes=3,ensemble_size=5,diffusion_steps=1,poke_mode='without',amp_dtype='none'),self.root/'results',step=1)
        records = json.loads((Path(summary['path'])/'metrics.json').read_text())
        self.assertEqual(records['scenes'][1]['pair_labels'],[10,30])
        files = list(Path(summary['path']).rglob('predictions.npz'))
        with np.load(files[0]) as a: self.assertEqual(a['target_definition'].item(),TARGET_DEFINITION)
        with self.assertRaisesRegex(ValueError,'counterfactual'):
            evaluate_validation(model,ds,ValidationConfig(num_scenes=3,visible_only=True,amp_dtype='none'))

    def test_full_evaluation_manifest_and_relative_archives(self):
        from finetuning_pair.evaluate import parse_args, prepare_run, evaluate_mode
        ds = self.ds(include_final_image=True)
        checkpoint = self.root/'base.pt'
        checkpoint.write_bytes(b'base checkpoint identity for manifest test')
        output = self.root/'evaluation'
        args = parse_args(['--pretrained','--checkpoint',str(checkpoint),
                           '--data-root',str(self.root),'--split','val','--output-dir',str(output),
                           '--device','cpu','--amp-dtype','none','--ensemble-size','5','--diffusion-steps','1'])
        info = dict(step=None,base_checkpoint={},lora_config=None,
                    train_config=dict(points_per_link=8,image_size=24,num_transitions=4,time_scale=4.,
                                      overlay_alpha=.85,target_definition=TARGET_DEFINITION))
        manifest = prepare_run(args,info,ds)
        self.assertEqual(len(manifest['settings']['pair_examples']),3)
        model = SpyModel()
        evaluate_mode(output,'without_poke',model,torch.device('cpu'),ds,manifest['settings'])
        self.assertEqual(model.given_pos.shape[1],1)  # last pair has one child query
        files = sorted((output/'without_poke'/'results').glob('*/results.npz'))
        self.assertEqual(len(files),3)
        for i,path in enumerate(files):
            with np.load(path) as saved:
                self.assertEqual(saved['target_definition'].item(),TARGET_DEFINITION)
                np.testing.assert_array_equal(saved['pair_labels'],ds[i]['pair_labels'].numpy())
                np.testing.assert_allclose(saved['image'],ds[i]['x'].numpy())
                np.testing.assert_array_equal(saved['rgb_frames'][0],saved['rgb_frames'][-1])
                np.testing.assert_array_equal(saved['final_image'],saved['image'])
        # Resuming completed pairs never samples new predictions.
        with patch.object(model,'predict_simulate',side_effect=AssertionError('already cached')):
            evaluate_mode(output,'without_poke',model,torch.device('cpu'),ds,manifest['settings'])
        self.assertEqual(prepare_run(args,info,ds)['settings']['target_definition'],TARGET_DEFINITION)
        changed = dict(manifest)
        changed['settings'] = dict(manifest['settings'],target_definition='old_actual_targets')
        (output/'run.json').write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,'different evaluation settings'):
            prepare_run(args,info,ds)

    def test_endpoint_rest_preserved_when_subsampling(self):
        ds = self.ds(num_transitions=2)
        torch.testing.assert_close(ds[1]['trajectories'][-1], self.ds()[1]['trajectories'][-1])
        torch.testing.assert_close(ds[1]['state_fractions'],torch.tensor([0.,.5,1.]))
        args = train_args(['--max-steps','2','--warmup-steps','0'])
        self.assertEqual(args.overlay_alpha,.85)
        self.assertEqual(args.val_poke_mode,'without')

    def test_reject_actual_target_checkpoints(self):
        path = self.root/'old.pt'
        torch.save({'format':'myriad_articulated_lora','version':1,'step':0,'train_config':{}},path)
        with self.assertRaises(ValueError): inspect_checkpoint(path)

    def test_sampling_cache_matches_fp32_and_autocast(self):
        from types import SimpleNamespace
        from myriad.model import FusedTransformerLayer
        from finetuning_pair.sampling import prepare_sampling_cache
        layer = FusedTransformerLayer(d_model=128,d_head=128,ff_expand=1)
        model = SimpleNamespace(transformer=SimpleNamespace(mid_level=nn.ModuleList([layer])))
        self.assertEqual(layer.kv_cache.k.dtype,torch.bfloat16)
        prepare_sampling_cache(model)
        self.assertEqual(layer.kv_cache.k.dtype,torch.float32)
        for name in ('k','v'):
            setattr(layer.kv_cache,name,getattr(layer.kv_cache,name).new_zeros((1,1,5,128)))
        with torch.no_grad():
            for start,length in ((0,3),(3,2)):
                output = layer(torch.randn(1,length,128),torch.zeros(1,length,1,32),torch.zeros(1,128),
                               block_mask=torch.ones(1,1,length,start+length,dtype=torch.bool),i_kv=start)
                self.assertTrue(torch.isfinite(output).all())
        with torch.autocast('cpu',dtype=torch.bfloat16):
            prepare_sampling_cache(model)
            self.assertEqual(layer.kv_cache.k.dtype,torch.bfloat16)
        prepare_sampling_cache(model)
        self.assertEqual(layer.kv_cache.k.dtype,torch.float32)

    def test_original_fm_head_forward_backward(self):
        from myriad.model import RFHeadDistributionOutput
        class OriginalHeadModel(SpyModel):
            def __init__(self):
                super().__init__()
                self.cond = nn.Linear(2,16)
                self.head = RFHeadDistributionOutput(depth=1,denoiser_width=16,d_cond=16)
            def forward(self, **kwargs):
                self.inputs = kwargs
                return self.head(self.cond(kwargs['pos']))
        model = OriginalHeadModel()
        batch = collate_pairs([self.ds()[1]])
        loss = training_loss(model,batch,amp_dtype='none')
        loss.backward()
        grads = [p.grad for p in model.head.parameters() if p.grad is not None]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))


if __name__ == '__main__': unittest.main()
