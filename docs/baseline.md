# Execution and archived experiment settings

## Inputs and runtime

Supply feature archives that you are authorized to use. The original working directory contains placeholders for these resources. The release copy has no experiment data, feature extraction recipe, split manifests or model weights. The paper describes AlexNet image features and Bag-of-Words text features. Verify the feature and row identity of any archive before evaluating against historical scores.

| Dataset argument | Files under `--DATA_ROOT` |
|---|---|
| `MIRFlickr` | `MIRFlickr/mir_train.mat`, `MIRFlickr/mir_query.mat`, `MIRFlickr/mir_database.mat` |
| `NUSWIDE` | `NUS-WIDE/nus_train.mat`, `NUS-WIDE/nus_query.mat`, `NUS-WIDE/nus_database.mat` |
| `COCO` | `MSCOCO/COCO_train.mat`, `MSCOCO/COCO_query.mat`, `MSCOCO/COCO_database.mat` |

Each split has image, text and label arrays. The expected keys are `I_tr/T_tr/L_tr`, `I_te/T_te/L_te`, and `I_db/T_db/L_db` for train, query, and database respectively. The COCO database uses HDF5 and the loader transposes its arrays. The other files use `scipy.io.loadmat`. Train shuffles paired rows and records their original IDs. Query and database preserve input order.

Training imports PyTorch, NumPy, SciPy, h5py, scikit-learn and Lightly, and uses CUDA. The loader casts feature and label arrays to float64, preserving the archived `np.float` conversion. Standalone evaluation uses the query and gallery files with a compatible saved checkpoint on CPU or CUDA. It requires no Lightly dependency or training controller. The publication-era package and device configuration remains unverified. [Tested environments](reproducibility_status.md) · [Evaluation commands](../README.md#standalone-evaluation)

## Archived settings

These values are the supplied launcher defaults.

| Preset | Bits | Batch | Epochs | Teacher LR | Image/text LR | K | a1/a2 | Clusters |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| MIRFlickr | 128 | 2048 | 60 | 0.001 | 0.0001 | 3000 | 0.6 / 0.6 | 30 |
| NUSWIDE | 128 | 1024 | 60 | 0.0001 | 0.0001 | 3300 | 0.5 / 0.5 | 15 |
| COCO | 16 | 2048 | 100 | 0.001 | 0.0005 | 3000 | 0.1 / 0.3 | 15 |

All presets use Adam, PyTorch seed 1, GPU 0, evaluation every 10 epochs, `lambda1=10`, `lambda2=1`, `beta=0.01`, and DCL temperature 0.07. The CLI keeps the original attention-diagnostic flags and `NUM_WORKERS` values. The actual loaders fix `num_workers=0`. See `sach/config.py` and `--help` for the full argument list.

`--DATA_ROOT` and `--LOG_DIR` default to `./datasets` and `./logs`, relative to the working directory. The launcher creates the log directory. `--MODEL_DIR` defaults to `./checkpoints`. When a new best is selected, the trainer saves into a dedicated `run-*` directory under that parent and logs the absolute path. Each entry point selects its preset even if `--DATASET` is overridden, consistent with the original dataset-specific launchers.

## Workflow

1. The launcher parses the preset, seeds PyTorch, selects the CUDA device and opens the directional-results log. The trainer loads the arrays and constructs the global coarse graph in original row order.
2. Each epoch runs the shuffled teacher pass. It gathers the coarse graph by sample ID, mixes neighbors into the paired features, clusters learned descriptors, propagates the multimodal graph and generates binary target codes. It keeps the codes and sample IDs for the hash-function pass.
3. The hash-function pass gathers features by the saved sample IDs. Each batch keeps the corresponding teacher-code positions and gathers both axes of the coarse affinity by the same IDs.
4. At the evaluation interval, the two heads encode query and gallery splits and score I→T and T→I with full-gallery MAP. Each strict improvement in the sum of these scores saves the two hash heads. The selection rule and evaluation arithmetic are unchanged.

The similarity operations stay with their consumers. `PCIRH.cal_similarity` builds the global coarse graph. `MyNet.forward` builds and propagates the cluster-enriched batch graph. The loss expressions stay in the two trainer passes, preserving their execution and optimizer order.

## Regression checks

Run `python -B -m unittest discover -s tests -v` from the release directory. Set `SACH_ORIGINAL` to the original source directory and `SACH_PHASE2A` to the approved refactor directory to enable comparisons with both snapshots. The tests use small fixtures. A focused test for real Lightly DCL runs if the package is installed. The teacher-pass parity test uses a differentiable substitute for DCL.

The unresolved research details include the exact source and feature files behind Table 1, run settings, NUS-WIDE batch size per published condition, checkpoint lineage, the effective historical evaluator and ranking ties, and the intended graph and loss formulations where the paper and source differ. [Implementation notes](implementation_notes.md) · [Known issues](known_issues.md)
