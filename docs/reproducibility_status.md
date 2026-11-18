# Reproducibility status

Phase 2C verified the Phase 2B implementation on bounded fixtures. Phase 2D verified that standalone evaluation returns the same full-gallery MAP as the Phase 2C evaluator on matched CPU and CUDA fixtures. The original feature archives, exact training environment and source-to-result records remain unavailable. No Table 1 condition has been independently reproduced.

## Data and artifacts

| Dataset | Published setup | Available inputs |
|---|---|---|
| MIRFlickr-25K | The paper retains 20,015 pairs, with 2,000 query and 18,015 gallery pairs. The 5,000 training pairs are drawn from the gallery. Features are 4,096-dimensional AlexNet and 1,386-dimensional BoW. | Historical logs and a loader contract. The exact MATLAB matrices and row identities are missing. |
| NUS-WIDE | The paper uses 186,577 pairs in its 10-concept subset, with 2,000 query and 184,577 gallery pairs. The 5,000 training pairs are drawn from the gallery. Features are 4,096-dimensional AlexNet and 1,000-dimensional BoW. | Historical logs and a loader contract. The matched runs do not identify their exact matrices or row identities. |
| MS COCO | Exploratory path in the snapshot. | The loader and logs are available. The exact split and feature archives are missing. |

The original `datasets/` and `chechpoints/` directories contained only placeholders. The release includes no original features, labels, raw media, training checkpoints, extraction script, exact AlexNet weights, BoW vocabulary, split IDs or raw Table 1 score exports. The paper describes the intended extraction and split sizes in §4.1. The loader expects precomputed matrices under `--DATA_ROOT`. [Loader contract](baseline.md) · [Published paper](https://doi.org/10.1016/j.neunet.2024.106211)

Before comparing benchmark numbers, obtain the exact feature files and split membership from the authors, record their hashes and confirm the applicable access and redistribution terms. An independently downloaded dataset with the same name can have different labels, feature extraction or partitions.

## Results and their sources

### A. Results reported by the paper

The paper's Table 1 SACH values are shown as **I→T / T→I**. Section 4.2 specifies full-gallery MAP. [Published paper](https://doi.org/10.1016/j.neunet.2024.106211)

| Condition | 16 bits | 32 bits | 64 bits | 128 bits |
|---|---|---|---|---|
| MIRFlickr-25K | 0.724 / 0.727 | 0.726 / 0.736 | 0.732 / 0.740 | 0.734 / 0.743 |
| NUS-WIDE | 0.612 / 0.614 | 0.624 / 0.621 | 0.627 / 0.625 | 0.633 / 0.630 |

The full-method MIRFlickr 64-bit T→I value differs between Table 1 (0.740) and Table 2 (0.743). The intended value requires author confirmation.

### B. Historical results supported by logs

Reparsing the preserved logs found **17 pairs of scores at three-decimal precision across seven of eight Table 1 conditions**. A match required the surrounding logged dataset and bit-length configuration as well as the directional pair. The counts below are evaluation entries. Log paths are relative to the original snapshot.

| Condition | Count | Logged batch sizes at matches | Matching log lines |
|---|---:|---|---|
| MIRFlickr 16 | 3 | 2048 | `logs/MIRFlickr_16.txt`, 14066, 14203, 16406 |
| MIRFlickr 32 | 0 | — | No matching pair in `logs/MIRFlickr_32.txt` |
| MIRFlickr 64 | 3 | 2048 | `logs/MIRFlickr_64.txt`, 1790, 1833, 1919 |
| MIRFlickr 128 | 4 | 2048 | `logs/MIRFlickr_128.txt`, 20155, 20199, 20242, 20371 |
| NUS-WIDE 16 | 4 | 512, 1024 and 2048 | `logs/NUSWIDE_16.txt`, 1205, 2635, 2854, 7235 |
| NUS-WIDE 32 | 1 | 1024 | `logs/NUSWIDE_32.txt`, 478 |
| NUS-WIDE 64 | 1 | 1024 | `logs/NUSWIDE_64.txt`, 515 |
| NUS-WIDE 128 | 1 | 1024 | `logs/NUSWIDE_128.txt`, 1051 |

The paper specifies NUS-WIDE batch size 2048. The corrected preset retains the archived setting of 1024. Rounded pairs can recur under different configurations. The historical output label and checkpoint banners supply no proof of the rank cutoff or successful file writes used for a specific run. The logs do not establish an exact source revision, dependency version, split identity or checkpoint for any published cell.

### C. Controlled unit and synthetic fixture results

The protected Phase 2A suite passed all 11 tests. With the Phase 2B executable files, the Phase 2C suite passed 27 tests and skipped the real Lightly check. Indexing fixtures compare distinguishable graph coordinates under shuffles and partial batches. The metric fixture returns identical AP of 1/53 for I→T and 1/51 for T→I across the three versions. Real PyTorch CPU checkpoint fixtures verify directory creation, independent runs, legacy loading, exact restoration and error reporting.

The teacher parity fixture substitutes a differentiable loss probe for Lightly and redirects explicit CUDA transfers to CPU. Actual PyTorch, scikit-learn clustering and matrix operations run in that fixture. The substitute checks training wiring and does not establish equivalence to real DCL.

### D. Results from the corrected implementation

An integration smoke test loaded generated MATLAB splits with 24 training, six query and 12 gallery rows, seven image features and five text features. A CPU wiring run used the DCL probe. A separate run in the existing CMH environment used a real RTX 4090, actual Lightly 1.4.3 and the corrected trainer without device redirection. With seed 202, four-bit codes and two training batches of 12, the real run completed one epoch and changed weights in the teacher and both hash heads. It saved both heads, restored exact parameter tensors into a fresh GPU controller and re-evaluated the splits. Synthetic full-gallery MAP was 0.37304142616642616 for I→T and 0.41737764550264544 for T→I. Both values and the precision/PR tensors were exactly equal before and after reload. All fixture files and checkpoints were temporary.

### E. Independently reproduced results on complete original datasets

**NOT TESTED.** No corrected full-dataset benchmark was run. The effect of sample-ID indexing on published MAP remains unknown.

## Environment and completed checks

| Component | Observed and status |
|---|---|
| Default CPU validation interpreter | **PASS** for the bounded CPU suite. Windows Python 3.10.9, PyTorch 1.12.1, NumPy 1.23.5, SciPy 1.10.0, scikit-learn 1.2.1 and h5py 3.7.0. PyTorch has no CUDA build. The Lightly import raises `ModuleNotFoundError: No module named 'lightly'` in this interpreter. |
| Existing CMH candidate | **PASS** for the focused runtime. The original IDE files name the `CMH` SDK. The matching local environment has Python 3.7.16, PyTorch 1.13.1+cu117, Torchvision 0.14.1+cu117, Lightly 1.4.3, NumPy 1.21.5 as imported, SciPy 1.7.3, scikit-learn 1.0.2 and h5py 3.8.0. The SDK reference provides no run-to-environment link for Table 1. |
| Lightly | **PASS** in CMH. The actual `lightly.loss.DCLLoss(temperature)` accepts the two paired graph-feature tensors and produces a differentiable scalar. Focused CPU and CUDA forward/backward checks at 0.07 produced finite scalar values and finite nonzero gradients for both inputs. |
| CUDA | **PASS** for the bounded CMH smoke. Actual execution used CUDA 11.7 and an NVIDIA GeForce RTX 4090 (driver 591.86). Initialization, training forwards, losses, backward, optimizer steps, checkpoint and both retrieval directions passed. Dataset-scale CUDA behavior is **NOT TESTED**. |
| Clean environment recreation | **NOT TESTED.** No new environment was created. The test reused installed packages. |
| Publication-era dependencies | **REQUIRES AUTHOR CONFIRMATION.** Recover dependency names and resolved versions, Python, CUDA runtime, seed controls and exact Lightly source revision. Phase 2D uses the equivalent float64 conversion without the archived NumPy alias. |

The installed Lightly 1.4.3 implementation L2-normalizes the input rows, uses paired positives, excludes diagonal comparisons from its two negative groups, adds separate within-view and cross-view log-sum-exp terms and averages the two directions. Its installed `dcl_loss.py` SHA-256 is `d15e02f21a03692e5e9be6e39d093f6dc3a3b6907eae9c0619331708cdb8ba6a`. The source in this release supplies only the temperature, with default 0.07. Lightly 1.4.3 is the tested version. The publication-era pin and minimum compatible version remain unknown. Lightly v1.5.21 lists a fix to DCL negative aggregation, so the shared API alone is insufficient to establish matching loss values across versions. [Lightly 1.4.3](https://pypi.org/project/lightly/1.4.3/) · [Lightly v1.5.21](https://github.com/lightly-ai/lightly/releases/tag/v1.5.21)

The Windows app sandbox denied SciPy's initial working-directory change into its installed DLL folder. During this verification process only, the import kept the workspace as its working directory for that permission error. SciPy performed its own absolute-path DLL loads. Normal directory handling was restored before SACH started. The GPU smoke used the actual installed scientific libraries, Lightly and SACH code. No packages, source files or persistent environment settings changed.

To rerun the corrected verification suite from this release directory under the Python 3.10 test interpreter, set the two protected comparison roots:

```powershell
$env:SACH_ORIGINAL = '<original source directory>'
$env:SACH_PHASE2A = (Resolve-Path -LiteralPath '../SACH_release').Path
python -Wall -B -m unittest discover -s tests -v
```

The test runner executes its optional real Lightly check when that package is installed in the test interpreter. This verification ran the actual DCL check and GPU smoke separately with CMH's Python 3.7. For a portable setup, validate a fresh isolated environment against the tested package versions after approval, then repeat the real DCL and bounded GPU checks before considering benchmark execution. [Implementation notes](implementation_notes.md) · [Known issues](known_issues.md)

## Phase 2D standalone evaluation

Fresh CPU processes exercised the MIRFlickr, NUS-WIDE and COCO evaluation split layouts with real saved hash-head checkpoints. Both directional scores matched the Phase 2C reference exactly on identical inputs. A distinct 53-item gallery fixture verified that relevant hits at ranks 53 and 51 contribute to I→T and T→I respectively.

A fresh evaluation process also ran on the RTX 4090 using the existing CMH interpreter. It matched the real Phase 2C GPU evaluator for the same checkpoint and synthetic query/gallery arrays. The standalone input root held only the evaluation files, and the checkpoint digest was unchanged. The run used the process-local SciPy import procedure described above. These are bounded fixtures. [Evaluation commands](../README.md#standalone-evaluation)
