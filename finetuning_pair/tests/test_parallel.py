"""Two-process CPU checks for the same collectives used by CUDA training."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from finetuning_pair.parallel import ParallelContext
from finetuning_pair.dataset import PairDataset, collate_pairs
from finetuning_pair.train import training_loss
from finetuning_pair.tests.test_pair_pipeline import make_scene, SpyModel

ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / '.test_artifacts'


def gradient_worker(rank, root):
    torch.set_num_threads(1)
    root = Path(root)
    ctx = ParallelContext(rank=rank, world_size=2, local_rank=rank)
    dist.init_process_group('gloo', init_method=f'file://{root / "gradient_rendezvous"}', rank=rank, world_size=2)
    try:
        ds = PairDataset(root,split='val',image_size=24,boundary_margin=0)
        samples = [ds[0],ds[1]]
        model = SpyModel()
        model.extra = nn.Parameter(torch.tensor(.3))
        model.unused = nn.Parameter(torch.tensor(.8))
        model.weight.data.fill_(.01+rank)
        ctx.synchronize_initial_parameters(model.parameters())
        assert model.weight.item() == torch.tensor(.01).item()
        for dummy in (False,True):
            model.zero_grad(set_to_none=True)
            batch = collate_pairs([samples[rank]])
            if dummy and rank == 1:
                batch['flow_loss_mask'].zero_()
                batch['motion_token_mask'].zero_()
            total = ctx.sum(batch['flow_loss_mask'].sum())
            loss = training_loss(model,batch,amp_dtype='none',loss_denominator=total)
            if rank == 1 and not dummy: loss = loss + .3*model.extra
            loss.backward()
            ctx.synchronize_gradients(model.parameters(),bucket_bytes=4)
            reference = SpyModel()
            expected_batch = collate_pairs(samples[:1] if dummy else samples)
            expected = training_loss(reference,expected_batch,amp_dtype='none')
            expected.backward()
            torch.testing.assert_close(model.weight.grad,reference.weight.grad)
            assert model.unused.grad is None
            if dummy: assert model.extra.grad is None
            else: torch.testing.assert_close(model.extra.grad,torch.tensor(.3))
        assert ctx.local_indices([7]) == (([7],False) if rank==0 else ([7],True))
        torch.manual_seed(123+rank)
        state = ctx.checkpoint_data_state(dict(epoch=1,offset=3,dataset_size=5))
        expected = torch.rand(4)
        if ctx.primary: torch.save(state,root/'rank_states.pt')
        ctx.barrier()
        restored = torch.load(root/'rank_states.pt',weights_only=True)
        torch.manual_seed(888)
        ctx.restore_rank_rng(restored)
        torch.testing.assert_close(torch.rand(4),expected)
        assert ctx.primary_flag(rank==0,torch.device('cpu'))
        (root/f'gradient_rank_{rank}.json').write_text(json.dumps(dict(weighted_gradients=True,dummy_shard=True,rank_rng_resume=True)))
    finally: ctx.close()


class TrainingSpy(SpyModel):
    def __init__(self):
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(()))
    def trainable_parameters(self): return self.parameters()
    def parameter_report(self): return {'trainable_parameters':2}
    def optimizer_groups(self, *, lora_lr, other_lr, weight_decay, lora_weight_decay):
        return [dict(params=[self.weight],lr=lora_lr,weight_decay=lora_weight_decay),
                dict(params=[self.unused],lr=other_lr,weight_decay=weight_decay)]


def schedule_worker(rank, root):
    from finetuning_pair import train
    from finetuning_pair.checkpoint_io import capture_rng_state, restore_rng_state
    torch.set_num_threads(1)
    root = Path(root)
    ctx = ParallelContext(rank=rank,world_size=2,local_rank=rank)
    dist.init_process_group('gloo',init_method=f'file://{root / "schedule_rendezvous"}',rank=rank,world_size=2)
    events = []
    def fake_save(path,model,optimizer,scheduler,step,base_reference,args,cursor,metrics,parallel=None):
        state = parallel.checkpoint_data_state(cursor.state_dict())
        if parallel.primary:
            payload = dict(format='myriad_articulated_lora',version=1,step=step,
                           train_config=train._resume_config(args),data_state=state,metrics=dict(metrics),
                           base_checkpoint={},lora_config={},target_modules=[],rng=capture_rng_state(),
                           model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict())
            torch.save(payload,path)
            events.append(['save',step,path.name])
    def fake_resume(path,model,optimizer,scheduler,**kwargs):
        payload = torch.load(path,weights_only=True)
        model.load_state_dict(payload['model'])
        optimizer.load_state_dict(payload['optimizer'])
        scheduler.load_state_dict(payload['scheduler'])
        if kwargs.get('restore_rng',True): restore_rng_state(payload['rng'])
        return payload['step']
    def fake_validation(model,dataset,config,**kwargs):
        assert rank==0
        step = len([e for e in events if e[0]=='validation'])
        events.append(['validation',2 if step==0 else 4])
        return dict(minADE_1=.5 if step==0 else .25)
    def fake_test(model,dataset,config,output,*,step):
        assert rank==0
        events.append(['test',step])
        folder=output/f'step_{step:06d}'
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'metrics.json').write_text(json.dumps(dict(step=step)))
        return dict(step=step,path=str(folder))
    try:
        with patch.dict(os.environ,dict(WORLD_SIZE='2',RANK=str(rank),LOCAL_RANK=str(rank))), \
             patch.object(train,'build_myriad',side_effect=TrainingSpy), \
             patch.object(train,'load_pretrained_weights',return_value=None), \
             patch.object(train,'FineTuningModel',side_effect=lambda bare,cfg:bare), \
             patch.object(train,'_save',side_effect=fake_save), \
             patch.object(train,'resume_training_checkpoint',side_effect=fake_resume), \
             patch.object(train,'evaluate_validation',side_effect=fake_validation), \
             patch.object(train,'run_test_subset',side_effect=fake_test):
            common = ['--data-root',str(root),'--output-dir',str(root/'run'),
                      '--cache-dir',str(root/'cache'),'--device','cpu',
                      '--max-steps','4','--warmup-steps','0','--batch-size','1',
                      '--image-size','24','--val-every-steps','2','--test-every-steps','2','--save-every-steps','1']
            args = train.parse_args(common+['--run-steps','2'])
            train._run_training(args,torch.device('cpu'),ctx)
            ctx.barrier()
            saved = torch.load(root/'run'/'latest.pt',weights_only=True)
            assert saved['step']==2 and saved['data_state']['world_size']==2
            args = train.parse_args(common+['--resume',str(root/'run'/'latest.pt')])
            train._run_training(args,torch.device('cpu'),ctx)
            ctx.barrier()
            final = torch.load(root/'run'/'latest.pt',weights_only=True)
            assert final['step']==4 and final['train_config']['world_size']==2
            (root/f'schedule_rank_{rank}.json').write_text(json.dumps(events))
    finally: ctx.close()


class ParallelTests(unittest.TestCase):
    def setUp(self):
        ARTIFACT_ROOT.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ARTIFACT_ROOT)
        self.root = Path(self.temp.name)
        make_scene(self.root,'val')
        for i in range(3): make_scene(self.root,'train',name=f'{i}.npz')
        make_scene(self.root,'test2')
    def tearDown(self): self.temp.cleanup()

    def test_auto_amp_requires_native_bf16(self):
        from finetuning_pair.train import _amp_setting
        from finetuning_pair.validation import _autocast_context
        with patch.object(torch.cuda,'is_bf16_supported',return_value=False) as supported:
            self.assertEqual(_amp_setting('auto',torch.device('cuda')),'none')
            supported.assert_called_with(including_emulation=False)
            self.assertEqual(type(_autocast_context(torch.device('cuda'),'auto')).__name__,'nullcontext')
            with self.assertRaises(ValueError): _amp_setting('bf16',torch.device('cuda'))

    def test_two_rank_weighted_gradients_padding_and_rng_resume(self):
        mp.spawn(gradient_worker,args=(str(self.root),),nprocs=2,join=True)
        for rank in range(2):
            result = json.loads((self.root/f'gradient_rank_{rank}.json').read_text())
            self.assertTrue(all(result.values()))

    def test_two_rank_periodic_validation_saves_tests_and_resume(self):
        mp.spawn(schedule_worker,args=(str(self.root),),nprocs=2,join=True)
        primary = json.loads((self.root/'schedule_rank_0.json').read_text())
        secondary = json.loads((self.root/'schedule_rank_1.json').read_text())
        self.assertEqual(secondary,[])
        self.assertEqual([e[1] for e in primary if e[0]=='validation'],[2,4])
        self.assertEqual([e[1] for e in primary if e[0]=='test'],[2,4])
        self.assertEqual([e[1] for e in primary if e[0]=='save' and e[2]=='latest.pt'],[1,2,3,4])
        metrics=[json.loads(line) for line in (self.root/'run'/'metrics.jsonl').read_text().splitlines()]
        self.assertEqual([m['step'] for m in metrics if m['event']=='checkpoint'],[1,2,3,4])
        self.assertTrue((self.root/'run'/'tests'/'step_000002'/'metrics.json').exists())
        self.assertTrue((self.root/'run'/'tests'/'step_000004'/'metrics.json').exists())


if __name__=='__main__': unittest.main()
