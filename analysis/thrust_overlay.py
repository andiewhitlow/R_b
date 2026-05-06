import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['text.usetex'] = False
import uproot, sys, os, glob
import awkward as ak

sys.path.append(os.path.expanduser("~/Desktop/R_b"))
from analysis.thrust import addthrustvariables

# load from ALL files
file_pattern = "/HEP/data/share/aleph/ntuples-withksloose/eventlevel/mc/output_qqb_*.root"
all_files = sorted(glob.glob(file_pattern))
print(f"Found {len(all_files)} files")

all_events = []
for fpath in all_files:
    f    = uproot.open(fpath)
    tree = f["events"]
    evts = tree.arrays(["Jets_px", "Jets_py", "Jets_pz"], entry_stop=2000)
    all_events.append(evts)
    print(f"  Loaded {len(evts)} events from {os.path.basename(fpath)}")

events = ak.concatenate(all_events)
print(f"Total events loaded: {len(events)}")

# collect up to N valid events
N = 200  # change this to plot more or fewer events
selected = []
for i in range(len(events)):
    px_i   = np.array(events[i]["Jets_px"])
    py_i   = np.array(events[i]["Jets_py"])
    pz_i   = np.array(events[i]["Jets_pz"])
    mags_i = np.sqrt(px_i**2 + py_i**2 + pz_i**2)
    if len(mags_i) >= 2 and np.all(mags_i > 5.0):
        selected.append(i)
    if len(selected) == N:
        break

print(f"Selected {len(selected)} valid events")

# compute thrust for all selected at once
batch = events[selected]
batch = addthrustvariables(batch)

# single 3D overlay plot
fig = plt.figure(figsize=(12, 10))
ax  = fig.add_subplot(111, projection='3d')

colors = ["#1f77b4", "#ff7f0e", "green", "purple"]

for plot_idx in range(len(selected)):
    evt_idx = selected[plot_idx]

    px = np.array(events[evt_idx]["Jets_px"])
    py = np.array(events[evt_idx]["Jets_py"])
    pz = np.array(events[evt_idx]["Jets_pz"])

    tx = float(batch["Event_thrust_x"][plot_idx])
    ty = float(batch["Event_thrust_y"][plot_idx])
    tz = float(batch["Event_thrust_z"][plot_idx])

    # normalize jet arrows to unit vectors
    for i in range(len(px)):
        mag = np.sqrt(px[i]**2 + py[i]**2 + pz[i]**2)
        if mag == 0:
            continue
        ux, uy, uz = px[i]/mag, py[i]/mag, pz[i]/mag
        col = colors[i % len(colors)]
        ax.quiver(0, 0, 0, ux, uy, uz,
                  color = col, alpha = 0.35,
                  arrow_length_ratio = 0.08, lw = 1.2)

    # thrust axis as thin transparent red line
    ax.plot([-tx, tx], [-ty, ty], [-tz, tz],
            color = 'red', alpha = 0.25, lw = 1.5)

# draw bold reference axes
ax.quiver(0, 0, 0, 1.2, 0, 0, color='black', lw=2, arrow_length_ratio=0.05)
ax.quiver(0, 0, 0, 0, 1.2, 0, color='black', lw=2, arrow_length_ratio=0.05)
ax.quiver(0, 0, 0, 0, 0, 1.2, color='black', lw=2, arrow_length_ratio=0.05)
ax.text(1.28, 0,    0,    'px', fontsize=10, color='black', fontweight='bold')
ax.text(0,    1.28, 0,    'py', fontsize=10, color='black', fontweight='bold')
ax.text(0,    0,    1.28, 'pz', fontsize=10, color='black', fontweight='bold')

ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_zlim(-1.3, 1.3)
ax.set_xlabel("px (normalized)", fontsize=9)
ax.set_ylabel("py (normalized)", fontsize=9)
ax.set_zlabel("pz (normalized)", fontsize=9)
ax.set_title(
    f"Normalized jet directions — {len(selected)} events overlaid\n"
    f"Blue=J0, Orange=J1  |  Red=thrust axis",
    fontsize=11
)
ax.view_init(elev=25, azim=50)
ax.grid(False)

plt.tight_layout()
outpath = os.path.expanduser("/users/awhitlo1/Desktop/R_b/analysis/output_test/thrust_overlay_3d.png")
plt.savefig("thrust_overlay.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved thrust_overlay.png")