import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# import seaborn
# import pandas as pd
# from matplotlib import pyplot as plt

matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size']=11

# final results
# old results were different, dont use
methods = ["M2IB","LC-IBA","LibraGrad","Fusion","Chefer"]

vdrop=[0.7899,1.0229,1.1326,1.5791,1.4623]
vincr = [47.147,41.442,38.849,35.892,35.477]

# text metrics
tdrop=[0.2732,0.4269,0.0168,0.1043,0.1589]
tincr = [28.060,28.683,1.971,18.672,27.282]

# maybe use np.array here
x=np.arange(len(methods))
width=0.36

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4.3))

# drop stuff
ax1.bar(x-width/2,vdrop,width,label="vdrop",color="#2980b9",alpha=0.9)
ax1.bar(x+width/2,tdrop,width,label="tdrop",color="#e67e22",alpha=0.9)

ax1.set_ylabel("Drop in confidence (lower is better)")
ax1.set_title("Drop Metrics")
ax1.set_xticks(x)
ax1.set_xticklabels(methods,rotation=20,ha="right")
ax1.legend(frameon=False)
ax1.grid(True,axis="y",alpha=0.3)

# old:
# ax1.set_ylim(...)

# increase metrics
ax2.bar(x-width/2,vincr,width,label="vincr",color="#2980b9",alpha=0.9)
ax2.bar(x+width/2,tincr,width,label="tincr",color="#e67e22",alpha=0.9)

ax2.set_ylabel("Increase in confidence (higher is better)")
ax2.set_title("Increase Metrics")
ax2.set_xticks(x)
ax2.set_xticklabels(methods,rotation=20,ha="right")
ax2.legend(frameon=False)
ax2.grid(True,axis="y",alpha=0.3)

# maybe tight layout was needed because labels
fig.tight_layout()

# save final
fig.savefig("clip_comparison.pdf",bbox_inches="tight")
print("  saved clip_comparison.pdf")

plt.close(fig)
print("Done.")
```
