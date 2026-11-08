# HRM External Force and Contact Location Estimation

Train Fx/Fy/Fz regression and contact-ID classification models from CSV inputs, then export test predictions to Excel and figures.
Python source files are in the project root, configurations are in `configs/`, and data are in `datasets/train` and `datasets/test`.
Run the commands below from the project root in the same Bash terminal.

## Download and install

Git, Git LFS, and `uv` are required. The complete CSV download is approximately **3.57 GB**.

```bash
git lfs install
git clone https://github.com/DaeyunJang/Hyper-Redundant-Manipulator-External-Force-Learning.git
cd Hyper-Redundant-Manipulator-External-Force-Learning
git lfs pull

bash scripts/setup_environment.sh
```

The setup script creates the Python 3.10 environment `env_hrm_force_estimation` and installs
PyTorch 2.7.1/CUDA 12.6 and the packages in `requirements.txt`. GPU execution requires a compatible NVIDIA driver.
Use `bash scripts/hrm_python.sh ...` for subsequent Python commands.

| Data path | CSV files | Size |
|---|---:|---:|
| `datasets/train/train1.csv` through `train40.csv` | 40 | Approx. 3.20 GB |
| `datasets/test/test1.csv` through `test18.csv` | 18 | Approx. 369.58 MB |

File numbers are not contact IDs. Labels come from the CSV column `contact_segment_id`.
Original filenames, contact IDs, and SHA256 hashes are listed in [datasets/file_names.csv](datasets/file_names.csv).
One complete dynamic recording per contact ID 1–18 is assigned to test: nine left-side and nine right-side recordings (seed 42).
Training uses the remaining 22 dynamic recordings and 18 static recordings.
The same original recording is not assigned to both train and test; CSV contents are preserved.
**Evaluate models trained from scratch on this split.** Do not reuse models trained on the earlier split.
If a CSV contains only a short LFS pointer, run `git lfs pull` again.

## Configure and check data

The default configuration is [configs/train.yaml](configs/train.yaml). Review these fields before running:

- `input_columns`, `force_columns`, `id_column`: CSV columns and their order.
- `source_force_unit`, `force_scope`, `ft_sensor_calibration`: force units, regression scope, and no-load bounds.
- `force_candidates`, `location_candidates`, `no_load_modes`: model architectures and no-load handling modes.
- `window_samples`, `validation_fraction`, `batch_size`, `max_epochs`: history length, validation split, batch size, and epoch limit.

Default inputs are four cable lengths, four tensions, and 18 relative angles.
The default force scope is `all_single_contact`, the input window contains 30 rows,
and validation uses the last 20% of numeric rows in each training CSV.
Original CSV files remain unchanged. Rows with missing or nonnumeric values in selected input/target columns are excluded.

```bash
# Check data and splits without training a model; a diagnostic output folder is created.
bash scripts/hrm_python.sh train.py --config configs/train.yaml --check-data

# Inspect the no-load bounds specified in YAML.
bash scripts/hrm_python.sh analyze_no_load.py --config configs/train.yaml
```

Select another input configuration with `--config`:

| Configuration file (under `configs/`) | Inputs |
|---|---|
| `train_cable_loadcell.yaml` | Cable lengths + tensions |
| `train_cable_angle.yaml` | Cable lengths + relative angles |
| `train_loadcell_angle.yaml` | Tensions + relative angles |
| `train_cable_loadcell_angle.yaml` | Cable lengths + tensions + relative angles |
| `train_cable_loadcell_estimated_pose.yaml` | Cable lengths + tensions + estimated tip pose |
| `train_cable_loadcell_fk_tip_tf.yaml` | Cable lengths + tensions + FK tip pose |

## Train

Start with one MLP force model, one MLP location model, and the `class0` mode for one epoch.
This command prepares the full configured dataset, trains the pair, and exports validation results to Excel.

```bash
HRM_RUN="run_$(date +%Y%m%d_%H%M%S)"
bash scripts/hrm_python.sh train.py --config configs/train.yaml \
  --models mlp --no-load-modes class0 --epochs 1 \
  --output "results/train/$HRM_RUN"
```

Run the following command separately for the complete comparison.
The default **11 force architectures × 11 location architectures × 2 modes = 242 combinations**,
with up to 60 epochs per model, require more time and storage than the single-pair check.

```bash
bash scripts/hrm_python.sh train.py --config configs/train.yaml \
  --output "results/train/${HRM_RUN}_all"
```

The training folder contains a `hrm_bundle.pt` for each pair, `validation.xlsx`, comparison tables, and the resolved configuration.
The prediction commands below use the first single-pair run. To use the full comparison instead,
set `HRM_RUN="${HRM_RUN}_all"`. Use a new output name for every run.

## Predict test data and organize Excel results

```bash
bash scripts/hrm_python.sh predict.py --train-dir "results/train/$HRM_RUN" \
  --output "results/predict/$HRM_RUN"

bash scripts/hrm_python.sh organize_results.py \
  "results/predict/$HRM_RUN/predictions.xlsx" --id-tolerance 2

bash scripts/hrm_python.sh analysis.py "results/predict/$HRM_RUN" \
  --output "results/analysis/$HRM_RUN"
```

`predict.py --train-dir` evaluates completed model pairs using their saved configuration.
It reads all CSV files currently in the saved `test_directory`; it does not replace model settings with the current YAML.
If you move a training folder to another computer, pass `--input-dir datasets/test` to specify the new test location.
Outputs include `predictions.xlsx`, `predictions.csv`, and `metrics.json`.

`organize_results.py` reads prediction Excel files and creates force/ID workbooks in a new adjacent `organized_*` folder.
`--id-tolerance 0` requires the exact ID; `2` allows a distance of ±2. Saved predictions are preserved.
`analysis.py` reads prediction folders and creates `force_and_id_all_combinations.xlsx` and comparison figures.
Neither step runs the models again. To compare multiple input configurations, pass their result folders together:

```bash
bash scripts/hrm_python.sh analysis.py results/predict/run_A results/predict/run_B
```

To run prediction and Excel organization in sequence, use this wrapper instead of the separate commands:

```bash
bash scripts/predict_and_organize.sh "results/train/$HRM_RUN" \
  "results/predict/${HRM_RUN}_organized" --id-tolerance 2
```

## MATLAB figures and main files

Open the project folder in MATLAB R2020b or later and run the desired script:

```matlab
run('analyze_matlab.m')  % Select force-estimation_*.xlsx produced by organize_results.py.
run('plot_analysis.m')   % Select force_and_id_all_combinations.xlsx produced by analysis.py.
```

Edit file paths, ID ranges, and figure options at the top of each script.
MATLAB reads saved Excel files and writes new files/folders next to the input workbook.

| File | Purpose |
|---|---|
| `train.py`, `model_zoo.py`, `data_utils.py` | Training, model definitions, and CSV processing |
| `predict.py` | Test prediction with saved models |
| `analyze_no_load.py` | No-load bounds and optional CSV statistics |
| `organize_results.py`, `analysis.py` | Excel organization and analysis |
| `analyze_matlab.m`, `plot_analysis.m` | MATLAB analysis and figures from Excel |
| `configs/*.yaml`, `requirements.txt`, `scripts/*.sh` | Configuration, dependencies, and environment/run tools |

All Python tools support `--help`.
