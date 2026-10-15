"""Numerical checks for cloud/axis alignment and whole-scene selection."""
import unittest
import numpy as np
from finetuning_pair.pointcloud_viewer import (
    SceneCase, axis_segment, backproject_cloud, rank_scenes, trajectory_image, JOINT_COLORS,
)


class PointCloudViewerTests(unittest.TestCase):
    def test_depth_backprojection_object_mask_and_world_extrinsic(self):
        rgb=np.full((2,3,3),128,dtype=np.uint8)
        depth=np.array([[2.,np.nan,1.],[0.,-1.,4.]])
        k=np.array([[2.,0,1],[0,2.,0],[0,0,1.]])
        e=np.eye(4);e[:3,3]=[1.,2.,3.]
        mask=np.array([[True,True,False],[True,True,True]])
        xyz,color=backproject_cloud(rgb,depth,k,e,mask)
        np.testing.assert_allclose(xyz,[[-2.,-2.,-1.],[1.,0.,1.]])
        np.testing.assert_allclose(color,np.full((2,3),128/255))

    def test_axis_segment_preserves_axis_line_and_axial_gauge(self):
        segment,center=axis_segment([0,0,-2],[1,2,300],[4,5,6],4)
        np.testing.assert_allclose(center,[1,2,6])
        np.testing.assert_allclose(np.cross(segment-[1,2,300],[0,0,1]),0)
        self.assertAlmostEqual(np.linalg.norm(segment[1]-segment[0]),4)

    def test_ranking_uses_all_scene_axes_and_rejects_missing_estimates(self):
        def row(scene,joint,error,gt='revolute'):
            return dict(scene_index=scene,joint_index=joint,mode='with_poke',gt_type=gt,axis_error_deg=error)
        rows=[row(0,0,1.),row(0,1,19.),row(1,0,3.),row(1,1,5.),
              row(2,0,.1),row(2,1,None),row(1,2,None,'fixed')]
        ranked=rank_scenes(rows,count=10)
        self.assertEqual([r['scene_index'] for r in ranked],[1,0])
        self.assertEqual(ranked[0]['mean_axis_error_deg'],4.)
        self.assertEqual(ranked[0]['eligible_joints'],3)

    def test_all_joint_trajectories_share_one_source_image(self):
        arrays=[];rows=[]
        for joint,origin in enumerate((.25,.7)):
            tracks=np.array([[[origin,.4]],[[origin+.05,.5]],[[origin+.1,.6]]])
            arrays.append(dict(initial_rgb=np.full((64,64,3),255,dtype=np.uint8),
                               image=np.zeros((3,64,64)),predictions=tracks[None]))
            rows.append(dict(joint_index=joint))
        image=trajectory_image(SceneCase(0,'with_poke',rows,arrays,{}),False,side=256)
        for color in JOINT_COLORS[:2]:
            target=np.array([int(255*x) for x in color])
            self.assertTrue(np.any(np.all(image==target,axis=-1)))


if __name__=='__main__':
    unittest.main()
