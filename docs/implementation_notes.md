# Phase 2B implementation notes

## Hash-function similarity target

`CustomDataSet` emits the original dataset row ID with each paired example. The training loader shuffles paired rows, so the teacher sees batches in shuffled order. `PCIRH.train_method` uses each batch's original IDs to gather its graph target. It collects B and `record_index` in the same sequence. The graph `S_cmm` remains indexed in original dataset order.

In the original and Phase 2A versions, the hash-function pass reordered both input modalities by `record_index` and used B in the collected order. It still sliced the graph with contiguous batch offsets. Those offsets described positions in the shuffled stream. Applying them to the graph selected unrelated dataset rows and columns, even when the target's shape exactly matched the predictions.

Phase 2B validates scalar integer IDs and their dataset range before feature indexing or optimizer activity. It requires a nonempty list with one ID per teacher-code row in collected order. For a batch whose IDs are `p`, feature rows and graph axes follow `p`, while the teacher codes follow the matching positions in the already-collected code array. Each target entry is `S_cmm[p[a], p[b]]`, including the final partial batch. All three cosine similarity comparisons receive this same target. The graph values, feature normalization, losses, coefficients and optimizers retain their previous definitions.

With distinguishable entries `S[u,v] = (8u + v)/64` and first-batch IDs [6, 0, 3], the old top-left target is `0/64` from position (0, 0), while the corrected target is `54/64` from sample pair (6, 6). The full two-axis matrix, normalized features and codes were checked against separate oracles. All three versions give the same head update for identity order. Shuffled fixtures retain the features and codes and produce different losses and head weights because the similarity target changes. The effect on published or benchmark-scale results is **unknown**.

## Checkpoint saving and loading

Every preset still selects a new best on strict improvement in the sum of I→T and T→I MAP. The selected head weights are saved for all presets, including MIRFlickr. On the first save the trainer creates a new `run-*` directory under `--MODEL_DIR`. It preserves the `<dataset>_<bits>bits.pth` basename and updates only that run's checkpoint at subsequent improvements.

The writer serializes `ImgNet` and `TxtNet` state dictionaries to a file in the same directory, flushes and syncs it, then publishes it with an atomic replacement. The trainer logs success and returns the absolute path after publication. Save failures propagate with context. Existing files in the parent and checkpoints from other runs are preserved.

Pass the logged absolute path to `trainer.load_checkpoints(path)`. This also accepts a legacy path relative to `MODEL_DIR`. Loading deserializes via CPU, validates both state dictionaries, and restores them strictly into the existing heads on their current devices. Use trusted checkpoint files, especially under older PyTorch versions that lack weights-only loading. The format stores inference weights and does not capture the state needed for training resume.

## Evaluation terminology

The retrieval evaluator ranks the full gallery using the existing signed-code distance and shared-label relevance. The compatibility function `calculate_top_map` still replaces its supplied cutoff with the actual gallery length. Training banners and directional metrics now read **MAP (full gallery)**. Ranking, AP calculation, averaging, ties, curve calculations and checkpoint selection arithmetic are unchanged.

## Research differences retained

The release keeps the observed graph construction, graph propagation, feature normalization, loss terms and weights, and teacher-then-hash-function pass order within each epoch. The NUS-WIDE preset retains batch size 1024. The paper and historical logs do not establish a single configuration for the published runs. Exact graph/normalization choices, loss definitions, scheduling and the per-condition NUS-WIDE batch size require author confirmation. The original dataset archives, historical checkpoints and source-to-result links remain unavailable.

Phase 2C verified the above implementation without source changes. Real Lightly 1.4.3 and a small real-CUDA corrected training/checkpoint/retrieval run passed in the existing CMH environment. The environment's link to the published runs and any benchmark effect remain unverified. [Paper-to-code mapping](paper_to_code_mapping.md) · [Reproducibility status](reproducibility_status.md)
