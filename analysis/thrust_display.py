import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['text.usetex'] = False
import uproot, sys, os
import awkward as ak

#find the analysis modules
sys.path.append(os.path.expanduser("~/Desktop/R_b"))
from analysis.thrust import addthrustvariables

#load events  
f = uproot.open("/HEP/data/share/aleph/ntuples-withksloose/eventlevel/mc/output_qqb_0.root")
tree = f["events"]
events = tree.arrays(["Jets_px", "Jets_py", "Jets_pz"], entry_stop=50000)

# collect up to 16 two-jet events
selected = []
for i in range(len(events)):
    px_i = np.array(events[i]["Jets_px"])
    pz_i = np.array(events[i]["Jets_pz"])
    py_i = np.array(events[i]["Jets_py"])
    
    # must have at least 2 jets with actual nonzero momentum
    mags_i = np.sqrt(px_i**2 + py_i**2 + pz_i**2)
    if len(mags_i) >= 2 and np.all(mags_i > 5.0):
        selected.append(i)
    if len(selected) == 16:
        break

print(f"Found {len(selected)} valid events")

# compute thrust for all selected at once
batch = events[selected]
batch = addthrustvariables(batch)

global_max = 0.0
for evt_idx in selected:
    px = np.array(events[evt_idx]["Jets_px"])
    py = np.array(events[evt_idx]["Jets_py"])
    pz = np.array(events[evt_idx]["Jets_pz"])
    event_max = float(np.max(np.sqrt(px**2 + py**2)))
    if event_max > global_max:
        global_max = event_max
global_max *= 1.3

# grid plot
ncols = 4
nrows = 4
fig = plt.figure(figsize=(20, 20))
colors = ["steelblue", "darkorange", "green", "purple"]

for plot_idx, evt_idx in enumerate(selected):
    ax = fig.add_subplot(nrows, ncols, plot_idx + 1, projection='3d')

    px = np.array(events[evt_idx]["Jets_px"])
    py = np.array(events[evt_idx]["Jets_py"])
    pz = np.array(events[evt_idx]["Jets_pz"])

    tx = float(batch["Event_thrust_x"][plot_idx])
    ty = float(batch["Event_thrust_y"][plot_idx])
    tz = float(batch["Event_thrust_z"][plot_idx])
    T  = float(batch["Event_thrust"]  [plot_idx])

    scale = global_max  # thrust axis line length

    # thrust axis line
    ax.quiver(0, 0, 0, tx*scale, ty*scale, tz*scale,
          color='red', arrow_length_ratio=0.1, lw=2, linestyle='--')
    ax.quiver(0, 0, 0, -tx*scale, -ty*scale, -tz*scale,
          color='red', arrow_length_ratio=0.1, lw=2, linestyle='--')

    # jets as 3D arrows from origin
    for i in range(len(px)):
        col  = colors[i % len(colors)]
        dot  = px[i]*tx + py[i]*ty + pz[i]*tz
        hemi = "+" if dot >= 0 else "-"

        ax.quiver(0, 0, 0,
                  px[i], py[i], pz[i],
                  color=col, arrow_length_ratio=0.15, lw=2)

        ax.text(px[i]*1.15, py[i]*1.15, pz[i]*1.15,
                f"J{i}({'+'  if dot>=0 else '-'})",
                color=col, fontsize=6)

    # axis limits from actual data
    ax.set_xlim(-global_max, global_max)
    ax.set_ylim(-global_max, global_max)
    ax.set_zlim(-global_max, global_max)

    ax.set_aspect("equal")
    ax.set_xlabel("px [GeV]", fontsize=7, labelpad=1)
    ax.set_ylabel("py [GeV]", fontsize=7, labelpad=1)
    ax.set_zlabel("pz [GeV]", fontsize=7, labelpad=1)
    ax.set_title(f"Jet momenta vs. Thrust Axis (event {evt_idx})", fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=5)

    # set a nice viewing angle
    ax.view_init(elev=30, azim=60)

plt.tight_layout()
outpath = os.path.expanduser("/users/awhitlo1/Desktop/R_b/analysis/output_test/thrust_display.png")
plt.savefig(outpath, dpi = 200, bbox_inches = "tight")
plt.close()
print(f"Saved to {outpath}")