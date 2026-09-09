"""Pre-fixed full-trajectory cases, all 25 channels and all ten samples."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('runs/phase23/tradeoff_v1');task=root/'task_development_full';out=root/'fixed_full_trajectories';out.mkdir(exist_ok=False)
    plan=json.loads(Path('runs/phase22/replicated_v1/evaluation_plan/manifest.json').read_text());manifest=json.loads((task/'manifest.json').read_text());records=[]
    for i in plan['fixed_visualization_indices']:
        y=np.load(manifest['sources'][i]['target']['path']);f,axes=plt.subplots(5,5,figsize=(18,12),sharex=True)
        for arm,color in [('C0','black'),('C1','tab:blue'),('C2','tab:orange')]:
            x=np.load(task/'exports'/f'seed123_{arm}'/f'{i:03d}.npy');assert x.shape==(10,len(y),25)
            for ch,ax in enumerate(axes.flat):
                for sample in x:ax.plot(sample[:,ch],color=color,alpha=.10,lw=.45)
                ax.plot(x[:,:,ch].mean(0),color=color,lw=.8,label=arm+' empirical sample mean')
        for ch,ax in enumerate(axes.flat):
            ax.plot(y[:,ch],color='tab:green',lw=.65,label='paired listener');ax.set_title(('AU'+str(ch+1)) if ch<15 else (['valence','arousal'][ch-15] if ch<17 else 'expression'+str(ch-17)),fontsize=9)
            ax.grid(alpha=.15)
        axes.flat[0].legend(fontsize=6);f.suptitle(f"Fixed Development index {i}, seed123, lambda .1 (C0=0), K10 bank0; full valid frames\n{manifest['sources'][i]['clip_id']} — faint lines: all 10 samples; solid: empirical average")
        f.supxlabel('Frame index (750-frame conditioning chunks; fixed global sample noise)');f.tight_layout(rect=(0,.02,1,.94));path=out/f'case_{i:03d}_all25.png';f.savefig(path,dpi=160);plt.close(f)
        records.append(dict(development_index=i,clip_id=manifest['sources'][i]['clip_id'],frames=len(y),figure=str(path),seed=123,selection='Phase22 pre-fixed visualization indices',all_channels=25,all_samples=10))
    (out/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
