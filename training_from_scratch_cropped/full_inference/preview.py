"""Render a saved example without loading the model or rerunning inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from .common import DEFAULT_OUT


def render(output, split="test1", pair=0, hypothesis=0):
    output = Path(output)
    manifest = json.loads((output/"run.json").read_text())
    row = json.loads((output/split/"records"/f"pair_{pair:05d}.json").read_text())
    with np.load(output/split/row["prediction_file"], allow_pickle=False) as arrays:
        prediction = arrays["predictions_source_pixels"][hypothesis]
        crop_prediction = arrays["predictions_model_pixels"][hypothesis]
        truth = arrays["ground_truth_source_pixels"]
        crop_truth = arrays["ground_truth_crop_normalized"]*(arrays["input_image"].shape[-1]-1)
        valid = arrays["valid"]
        image = (arrays["input_image"].transpose(1,2,0)+1)/2
        top,left,side,size = arrays["crop_bounds"].tolist()
    with np.load(output/split/row["source_archive"], allow_pickle=False) as arrays:
        source = arrays["rgb"][0]
    fig, axes = plt.subplots(1,2,figsize=(11,5), constrained_layout=True)
    for ax, rgb, gt, pred in ((axes[0],source,truth,prediction), (axes[1],image,crop_truth,crop_prediction)):
        ax.imshow(rgb)
        for track in range(valid.shape[1]):
            points = gt[:,track].copy(); points[~valid[:,track]] = np.nan
            ax.plot(points[:,0],points[:,1],color="#28e071",lw=1.6,alpha=.85)
            ax.plot(pred[:,track,0],pred[:,track,1],color="#ff3fcf",ls="--",lw=1.3,alpha=.85)
            ax.scatter(gt[0,track,0],gt[0,track,1],c="white",s=12,edgecolors="black",linewidths=.4)
        ax.set_xlim(-.5,rgb.shape[1]-.5); ax.set_ylim(rgb.shape[0]-.5,-.5)
        ax.set_axis_off()
    axes[0].add_patch(Rectangle((left-.5,top-.5),side,side,fill=False,ec="cyan",lw=1.5))
    axes[0].set_title("Initial source RGB and crop boundary")
    axes[1].set_title("Exact cropped model input")
    fig.legend(handles=[Line2D([0],[0],color="#28e071",label="GT relative trajectory"),
        Line2D([0],[0],color="#ff3fcf",ls="--",label=f"Predicted hypothesis {hypothesis}")],
        loc="outside lower center",ncol=2)
    fig.suptitle(f"{split} pair {pair} · joint {row['joint_ordinal']} · step {manifest['checkpoint_step']}",fontsize=12)
    target = output/"previews"/f"{split}_pair_{pair:05d}_hypothesis_{hypothesis}.png"
    target.parent.mkdir(exist_ok=True)
    fig.savefig(target,dpi=160); plt.close(fig)
    print(target)
    return target


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--split",choices=("test1","val"),default="test1")
    parser.add_argument("--pair",type=int,default=0)
    parser.add_argument("--hypothesis",type=int,default=0)
    args=parser.parse_args()
    render(args.output,args.split,args.pair,args.hypothesis)
