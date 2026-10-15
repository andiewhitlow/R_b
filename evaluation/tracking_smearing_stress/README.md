# Tracking-smearing stress test

This directory contains the inference and plotting workflow used to measure how
the Rb single- and double-tag efficiencies respond to tracking smearing.

The repository includes the inference support, smearing cards, current tag
definitions, the constituent-based thrust calculation, and plotting code. Large ROOT
files, ONNX model files, nominal score pickles, and generated outputs are not
stored in Git.

## Setup

Clone the repository and enter this directory:

```bash
git clone https://github.com/andiewhitlow/R_b.git
cd R_b/evaluation/tracking_smearing_stress
python -m pip install -r requirements.txt
```

`evaluation/inference.py -h` should list the tracking-variation arguments.

## Run the stress inference

Run this on a batch worker or interactive allocation, not an lxplus login node:

```bash
python run_stress_inference.py \
  --analysis-dir ../.. \
  --mc-dir /path/to/eventlevel/mc \
  --model /path/to/model.onnx \
  --preprocess /path/to/preprocess.json \
  --translation ../translations/translations.json \
  --output-dir /path/to/tracking_stress_scores \
  --strengths 100 200 300 \
  --workers 2
```

The numeric strengths are internal names for the standard, twice-correction,
and three-times-correction cards. The physical SIP2D, SIP3D, dxy, and dz values
are written on the plots. The runner is resumable and skips valid outputs.

## Make the plots

```bash
python make_heatmaps.py \
  --mc-dir /path/to/eventlevel/mc \
  --nominal-score-dir /path/to/output_scores_model_20260305_withnewks_standardized \
  --stress-score-dir /path/to/tracking_stress_scores \
  --thrust-script ./thrust.py \
  --output-dir /path/to/tracking_stress_results
```

For a five-file preview, append `--limit-files 5`.

The analyzer produces single- and double-tag efficiency tables, migration
tables, b/c/uds heatmaps, a migration summary plot, and a selection audit.

## Analysis definition

The current working points are:

```text
Q: B > 0.855
S: 0.601 < B <= 0.855
L: 0.396 < B <= 0.601
C: B <= 0.396 and C > 0.605
X: B <= 0.396 and C <= 0.524
U: otherwise
```

Thrust is calculated from jet constituents before jet selection. The event and jet requirements are
implemented in `make_heatmaps.py`; exactly two selected jets are required for
the efficiency calculation. UU events and U-containing displayed categories
are omitted from the reported tag tables and heatmaps.

`oscar_job_template.sbatch` is an example Slurm submission script. On lxplus,
wrap the same commands in the collaboration's normal HTCondor submission.
