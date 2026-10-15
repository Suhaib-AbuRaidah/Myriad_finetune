"""Live Open3D RGB-D viewer with all scene joints and 2D trajectories."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from finetuning_pair.paths import artifact_path

JOINT_COLORS = ((.95, .18, .08), (.18, .45, 1.), (.95, .56, .04), (.8, .18, .8), (.05, .7, .65))


@dataclass
class SceneCase:
    scene_index: int
    mode: str
    records: list[dict]
    arrays: list[dict]
    references: dict
    rank: int = 0


def rank_scenes(records, mode="with_poke", count=10):
    """Rank whole scenes; require metrics for every eligible GT revolute joint."""
    grouped = {}
    for row in records:
        if row["mode"] == mode:
            grouped.setdefault(row["scene_index"], []).append(row)
    ranked = []
    for scene, rows in grouped.items():
        revolute = [r for r in rows if r["gt_type"] == "revolute"]
        if not revolute or any(r.get("axis_error_deg") is None or not math.isfinite(r["axis_error_deg"]) for r in revolute):
            continue
        ranked.append(dict(scene_index=scene, mode=mode,
                           mean_axis_error_deg=float(np.mean([r["axis_error_deg"] for r in revolute])),
                           max_axis_error_deg=max(r["axis_error_deg"] for r in revolute),
                           eligible_joints=len(rows), revolute_joints=len(revolute),
                           joint_indices=[r["joint_index"] for r in sorted(rows,key=lambda r:r["joint_index"])],
                           per_joint_axis_error_deg={str(r["joint_index"]):r.get("axis_error_deg") for r in rows}))
    ranked.sort(key=lambda r:(r["mean_axis_error_deg"],r["scene_index"]))
    return [dict(rank=i+1, **row) for i,row in enumerate(ranked[:count])]


def load_cases(run, mode="with_poke", top_k=None, scene_index=None):
    run = Path(run)
    records = [json.loads(p.read_text()) for p in sorted((run / "records").glob("*.json"))]
    references = json.loads((run / "geometry_reference.json").read_text()) if (run / "geometry_reference.json").exists() else {}
    ranking = rank_scenes(records, mode, top_k or 10)
    if top_k:
        if not ranking:
            raise ValueError("No scenes have axis errors for all eligible revolute joints")
        scenes = [(r["scene_index"],r["rank"]) for r in ranking]
    else:
        scenes = [(0 if scene_index is None else scene_index,0)]
    cases = []
    for scene, rank in scenes:
        rows = sorted([r for r in records if r["mode"] == mode and r["scene_index"] == scene and not r.get("skipped")],
                      key=lambda r:r["joint_index"])
        if not rows:
            raise ValueError(f"No saved {mode} predictions for scene {scene}")
        arrays = []
        for row in rows:
            with np.load(run / "predictions" / f"{row['pair_index']:04d}_{mode}.npz", allow_pickle=False) as a:
                arrays.append({k:a[k] for k in a.files})
        cases.append(SceneCase(scene,mode,rows,arrays,references,rank))
    return cases, ranking


def backproject_cloud(rgb, depth, intrinsic, extrinsic, mask):
    """Pixel-center +Z depth to world points; mask removes non-object pixels."""
    rgb, depth, mask = np.asarray(rgb),np.asarray(depth),np.asarray(mask,dtype=bool)
    if depth.ndim != 2 or rgb.shape != (*depth.shape,3) or mask.shape != depth.shape:
        raise ValueError("RGB, depth and object mask dimensions must match")
    intrinsic,extrinsic = np.asarray(intrinsic,dtype=float),np.asarray(extrinsic,dtype=float)
    if intrinsic.shape != (3,3) or extrinsic.shape != (4,4) or not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all():
        raise ValueError("Expected finite 3x3 intrinsic and 4x4 T_camera_world matrices")
    valid = mask & np.isfinite(depth) & (depth > 0)
    y,x = np.nonzero(valid)
    if not len(x):
        raise ValueError("Object mask has no finite positive depth pixels")
    rays = np.column_stack((x,y,np.ones(len(x)))) @ np.linalg.inv(intrinsic).T
    camera = rays * (depth[y,x]/rays[:,2])[:,None]
    points = np.column_stack((camera,np.ones(len(x)))) @ np.linalg.inv(extrinsic).T
    return points[:,:3],rgb[y,x].astype(float)/255.


def case_cloud(case):
    a = case.arrays[0]
    mask = np.isin(a["initial_segmentation"],a["link_indices"])
    return backproject_cloud(a["initial_rgb"],a["initial_depth"],a["camera_intrinsic"],a["camera_extrinsic"],mask)


def trajectory_image(case, ground_truth=True, side=580):
    """Overlay all pair forecasts on one original initial image, in source pixels."""
    rgb = case.arrays[0]["initial_rgb"]
    h,w = rgb.shape[:2]
    image = Image.fromarray(rgb).resize((side,side),Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(image)
    for j,(row,a) in enumerate(zip(case.records,case.arrays)):
        color = tuple(int(255*x) for x in JOINT_COLORS[j%len(JOINT_COLORS)])
        size = a["image"].shape[-1]
        crop = min(h,w)
        offset = np.array([round((w-crop)/2),round((h-crop)/2)])
        def pixels(normalized):
            source = ((normalized*(size-1)+.5)*crop/size-.5)+offset
            return source * [side/w,side/h]
        pred = pixels(a["predictions"][0])
        truth = pixels(a["ground_truth"]) if ground_truth and "ground_truth" in a else None
        valid = a.get("valid",np.ones(pred.shape[:2],dtype=bool))
        for point in range(pred.shape[1]):
            if truth is not None:
                for state in range(1,len(truth)):
                    if valid[state-1,point] and valid[state,point]:
                        draw.line([tuple(truth[state-1,point]),tuple(truth[state,point])],fill=(0,170,70),width=2)
            for state in range(1,len(pred)):
                segment = pred[state-1:state+1,point]
                if np.isfinite(segment).all():
                    draw.line([tuple(x) for x in segment],fill=color,width=2)
            for p in pred[:,point]:
                if np.isfinite(p).all():
                    x,y = p
                    draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
        draw.text((10,10+18*j),f"Joint index {row['joint_index']}: prediction",fill=color,stroke_width=1,stroke_fill="white")
    if ground_truth and "ground_truth" in case.arrays[0]:
        draw.text((10,side-22),"Green: ground truth tracks",fill=(0,130,50),stroke_width=1,stroke_fill="white")
    return np.asarray(image)


def axis_segment(axis,pivot,anchor,length):
    axis,pivot,anchor = np.asarray(axis,dtype=float),np.asarray(pivot,dtype=float),np.asarray(anchor,dtype=float)
    if not np.isfinite(axis).all() or not np.isfinite(pivot).all() or np.linalg.norm(axis)<1e-12:
        raise ValueError("Axis/pivot must be finite with a nonzero axis")
    direction = axis/np.linalg.norm(axis)
    center = pivot+direction*((anchor-pivot)@direction)
    return np.array([center-.5*length*direction,center+.5*length*direction]),center


def axis_mesh(o3d,axis,pivot,anchor,length,color):
    endpoints,center = axis_segment(axis,pivot,anchor,length)
    direction=(endpoints[1]-endpoints[0])/length
    z=np.array([0.,0.,1.]); cross=np.cross(z,direction); sine=np.linalg.norm(cross); cosine=z@direction
    if sine < 1e-12:
        rotation=np.eye(3) if cosine>0 else np.diag([1.,-1.,-1.])
    else:
        v=cross/sine
        skew=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        rotation=np.eye(3)+sine*skew+(1-cosine)*(skew@skew)
    mesh=o3d.geometry.TriangleMesh.create_cylinder(radius=length*.006,height=length,resolution=16)
    mesh.rotate(rotation,center=(0,0,0));mesh.translate(center)
    mesh.paint_uniform_color(color);mesh.compute_vertex_normals()
    return mesh


def scene_geometries(case,ground_truth=True,stored_reference=False):
    import open3d as o3d
    xyz,colors=case_cloud(case)
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    cloud.colors=o3d.utility.Vector3dVector(colors)
    scale=max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-3)
    geometries=[("Object RGB-D",cloud)]
    for j,(row,a) in enumerate(zip(case.records,case.arrays)):
        anchor=a["reference_world"].mean(0)
        color=JOINT_COLORS[j%len(JOINT_COLORS)]
        axis,pivot=row.get("estimated_axis_world"),row.get("estimated_pivot_world")
        if axis is not None and pivot is None and row.get("predicted_type") == "prismatic":
            pivot=anchor  # Translation direction has no identifiable axis location.
        if axis is not None and pivot is not None:
            geometries.append((f"Predicted joint {row['joint_index']}",axis_mesh(o3d,axis,pivot,anchor,.7*scale,color)))
        if ground_truth and row.get("gt_type") == "revolute":
            ref=case.references.get(str(row.get("pair_index")),{})
            gt_axis=ref.get("kinematic_axis_world",row["gt_axis_world"])
            gt_pivot=ref.get("kinematic_pivot_world",row["gt_pivot_world"])
            geometries.append((f"GT joint {row['joint_index']}",axis_mesh(o3d,gt_axis,gt_pivot,anchor,.72*scale,(.05,.75,.2))))
            if stored_reference:
                geometries.append((f"Stored GT joint {row['joint_index']}",axis_mesh(o3d,row["gt_axis_world"],row["gt_pivot_world"],anchor,.74*scale,(0.,.8,.9))))
    return geometries,cloud.get_axis_aligned_bounding_box()


class JointSceneViewer:
    """Live Open3D window and paired Tk window showing every joint's tracks."""
    def __init__(self,cases,output_dir,test_seconds=0):
        import open3d as o3d
        import tkinter as tk
        from PIL import ImageTk
        self.o3d,self.tk,self.ImageTk=o3d,tk,ImageTk
        self.cases,self.index,self.output_dir=cases,0,Path(output_dir)
        self.output_dir.mkdir(parents=True,exist_ok=True)
        self.test_seconds,self.running=test_seconds,True
        self.root=tk.Tk()
        self.root.title("MYRIAD | ALL joint trajectories and controls")
        self.root.geometry("610x900+970+30")
        self.root.protocol("WM_DELETE_WINDOW",self.close)
        self.title=tk.Label(self.root,font=("sans",13,"bold"));self.title.pack(pady=8)
        nav=tk.Frame(self.root);nav.pack()
        for label,callback in (("Previous (P)",lambda:self.navigate(-1)),("Next (N)",lambda:self.navigate(1)),
                               ("Reset (R)",self.reset_camera),("Snapshot (S)",self.save_snapshot)):
            tk.Button(nav,text=label,command=callback).pack(side=tk.LEFT,padx=3)
        self.gt=tk.BooleanVar(value=True)
        tk.Checkbutton(self.root,text="Show transform-consistent GT axes / tracks (green)",
                       variable=self.gt,command=lambda:self.update(False)).pack(anchor="w",padx=12)
        self.stored=tk.BooleanVar(value=False)
        tk.Checkbutton(self.root,text="Also show stored GT axes (cyan)",
                       variable=self.stored,command=lambda:self.update(False)).pack(anchor="w",padx=12)
        self.image=tk.Label(self.root);self.image.pack()
        self.info=tk.Label(self.root,justify=tk.LEFT,font=("sans",11));self.info.pack(anchor="w",padx=15,pady=8)
        tk.Label(self.root,justify=tk.LEFT,text="Drag point cloud to rotate | wheel to zoom\n"
                 "Joint colors match predicted axes and tracks; green is GT.\n"
                 "Cloud: initial visible object surface.\n"
                 "Each pair's trajectory freezes its own parent.\n"
                 "Close either window to finish.").pack(anchor="w",padx=15)
        self.vis=o3d.visualization.VisualizerWithKeyCallback()
        if not self.vis.create_window("MYRIAD | ALL predicted object joint axes",940,900,20,30,visible=True):
            self.root.destroy()
            raise RuntimeError("Open3D could not open a window; run from your desktop with DISPLAY set")
        options=self.vis.get_render_option()
        options.background_color=np.array([.94,.95,.97]);options.point_size=4.;options.light_on=False
        def key(callback):
            def invoke(vis):
                callback()
                return False
            return invoke
        for letter,callback in (('N',lambda:self.navigate(1)),('P',lambda:self.navigate(-1)),
                                 ('R',self.reset_camera),('S',self.save_snapshot)):
            self.vis.register_key_callback(ord(letter),key(callback))
        self.update(True)

    def close(self):
        self.running=False

    def navigate(self,delta):
        self.index=(self.index+delta)%len(self.cases)
        self.update(True)

    def reset_camera(self):
        case=self.cases[self.index]
        center=self.bounds.get_center()
        world_camera=np.linalg.inv(case.arrays[0]["camera_extrinsic"].astype(float))
        front=world_camera[:3,3]-center
        front/=max(float(np.linalg.norm(front)),1e-8)
        view=self.vis.get_view_control()
        view.set_lookat(center);view.set_front(front);view.set_up(-world_camera[:3,1]);view.set_zoom(.7)

    def save_snapshot(self):
        case=self.cases[self.index]
        folder=self.output_dir/f"scene_{case.scene_index:03d}_{case.mode}"
        path=folder/"pointcloud_all_axes.png"
        self.vis.capture_screen_image(str(path),do_render=True)
        view=Image.open(path).convert("RGB")
        tracks=Image.fromarray(trajectory_image(case,self.gt.get()))
        tracks=tracks.resize((view.height,view.height),Image.Resampling.LANCZOS)
        combined=Image.new("RGB",(view.width+tracks.width,view.height),"white")
        combined.paste(view,(0,0));combined.paste(tracks,(view.width,0))
        combined.save(folder/"pointcloud_and_all_trajectories.png")
        print(f"Snapshot: {folder}",flush=True)

    def update(self,reset):
        case=self.cases[self.index]
        self.title.config(text=f"Scene {case.scene_index} | {case.mode} | {self.index+1}/{len(self.cases)}")
        self.vis.clear_geometries()
        self.geometries,self.bounds=scene_geometries(case,self.gt.get(),self.stored.get())
        for index,(name,geometry) in enumerate(self.geometries):
            self.vis.add_geometry(geometry,reset_bounding_box=index==0)
        image=trajectory_image(case,self.gt.get(),side=560)
        self.photo=self.ImageTk.PhotoImage(Image.fromarray(image))
        self.image.config(image=self.photo)
        lines=[]
        for row in case.records:
            error=row.get("axis_error_deg")
            annotation=f"axis error {error:.3f} deg" if error is not None else f"type {row.get('predicted_type','unknown')}"
            lines.append(f"Joint {row['joint_index']}: {annotation}")
        self.info.config(text="\n".join(lines))
        folder=self.output_dir/f"scene_{case.scene_index:03d}_{case.mode}"
        folder.mkdir(exist_ok=True)
        Image.fromarray(image).save(folder/"all_joint_trajectories.png")
        if reset:self.reset_camera()
        self.vis.update_renderer()
        print(f"Showing scene {case.scene_index}: {len(case.records)} joints, {len(self.geometries)-1} axis overlays",flush=True)

    def run(self):
        start=time.monotonic();captured=False
        try:
            while self.running and self.vis.poll_events():
                self.root.update_idletasks();self.root.update()
                self.vis.update_renderer()
                if self.test_seconds and time.monotonic()-start>1 and not captured:
                    self.save_snapshot();captured=True
                if self.test_seconds and time.monotonic()-start>self.test_seconds:
                    break
                time.sleep(.01)
        finally:
            self.vis.destroy_window()
            self.root.destroy()


def show_saved_run(run,mode="with_poke",top_k=None,scene_index=None,output_dir=None,list_only=False,test_seconds=0):
    run=artifact_path(run)
    out=artifact_path(output_dir) if output_dir else run/"open3d_visualizations"
    out.mkdir(parents=True,exist_ok=True)
    cases,ranking=load_cases(run,mode,top_k,scene_index)
    for case in cases:
        folder=out/f"scene_{case.scene_index:03d}_{case.mode}"
        folder.mkdir(exist_ok=True)
        Image.fromarray(trajectory_image(case)).save(folder/"all_joint_trajectories.png")
    if top_k:
        (out/f"top_{top_k}_{mode}_scenes.json").write_text(json.dumps(ranking,indent=2)+"\n")
        with (out/f"top_{top_k}_{mode}_scenes.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=("rank","scene_index","mode","mean_axis_error_deg","max_axis_error_deg","eligible_joints","revolute_joints"),extrasaction="ignore")
            writer.writeheader();writer.writerows(ranking)
        for r in ranking:
            print(f"#{r['rank']:2d} scene {r['scene_index']:2d}: mean axis {r['mean_axis_error_deg']:.4f} deg; "
                  f"max {r['max_axis_error_deg']:.4f}; {r['eligible_joints']} joints",flush=True)
    if not list_only:
        JointSceneViewer(cases,out,test_seconds).run()
    return ranking


def show_current_prediction(rgb,depth,mask,k,e,image,predictions,known,output_dir,test_seconds=0):
    """Fit and display the selected raw-input pair in the current RGB-D frame."""
    from finetuning_pair.analyze_joints import fit_joint,clean
    from finetuning_pair.dataset import _backproject
    from finetuning.joint_parame_estim.run import from_normalized,unit
    h,w=depth.shape
    size=image.shape[-1]
    pixels=from_normalized(known[0],h,w,size)
    x,y=np.rint(pixels).astype(int).T
    if np.any(x<0) or np.any(x>=w) or np.any(y<0) or np.any(y>=h):
        raise ValueError("Query pixels are outside the current depth image")
    point_depth=depth[y,x]
    if not np.all(np.isfinite(point_depth)&(point_depth>0)):
        raise ValueError("Every query point needs finite positive current depth for 3D joint fitting")
    xyz=_backproject(pixels,point_depth,k,e)
    states=np.arange(2,predictions.shape[1])
    if not len(states):
        raise ValueError("3D joint fitting needs predicted states after state 1")
    normalized=predictions[0,states]
    valid=np.isfinite(normalized).all(-1)&((normalized>=0)&(normalized<=1)).all(-1)
    fitted=fit_joint(xyz,from_normalized(predictions[0],h,w,size)[states],valid,k,e)
    if fitted is None:
        raise ValueError("Too few valid predicted point observations to estimate a 3D joint")
    kind=fitted['kind'];params=fitted['fits'][kind]['params']
    record=clean(dict(scene_index=0,joint_index=0,mode='current_rgbd',predicted_type=kind,gt_type='unknown',
                      estimated_axis_world=unit(params[:3]) if kind!='fixed' else None,
                      estimated_pivot_world=params[3:6] if kind=='revolute' else None,
                      fits=fitted))
    out=artifact_path(output_dir)
    (out/'joint_estimate.json').write_text(json.dumps(record,indent=2)+'\n')
    arrays=dict(initial_rgb=rgb,initial_depth=depth,initial_segmentation=np.where(mask,0,-2),link_indices=np.array([0]),
                camera_intrinsic=k,camera_extrinsic=e,image=image,predictions=predictions,reference_world=xyz,
                valid=np.isfinite(predictions[0]).all(-1),known_states=np.array(len(known)))
    np.savez_compressed(out/'pointcloud_inputs.npz',**arrays)
    JointSceneViewer([SceneCase(0,'current_rgbd',[record],[arrays],{})],out,test_seconds).run()
