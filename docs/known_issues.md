# Remaining implementation and reproducibility issues

Phase 2B corrects sample-ID indexing, checkpoint saving and the full-gallery MAP label. Phase 2D adds working standalone evaluation. The remaining issues require separate engineering work or research confirmation. [Implementation notes](implementation_notes.md) · [Evaluation commands](../README.md#standalone-evaluation)

| Area | Observed behavior and impact |
|---|---|
| Training resume | Checkpoints contain the two hash heads. Resuming a training trajectory would also require optimizer, teacher and RNG state. |
| Graph and losses | Rank pruning, compact-feature clustering and fusion, degree scaling, normalization, reconstruction/contrastive losses and the training schedule differ from the paper at the points identified in Phase 1.1. Author confirmation is required to choose a publication configuration. |
| Attention diagnostics | Two attention reconstruction values are logged outside the backpropagated objective. Their forwards can still consume random numbers and change BatchNorm buffers. The inactive model-level attention modules also consume RNG at initialization. Both are retained for parity. |
| Cluster category handling | The text-fitted one-hot encoder also transforms image cluster IDs. This can fail if an image label is absent in the observed text categories. |
| Baseline environment and records | PyTorch is seeded, but the full clustering/environment randomness is unverified. Original features and checkpoints are absent, and rounded historical MAP values do not identify their source revision or full run configuration. |

The published benchmark scores remain historically reported values. A new fit after an objective or configuration change needs its own result record.
