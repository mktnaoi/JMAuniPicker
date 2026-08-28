# models02_preprint_v260416 — preprint version (superseded)

**These are the model weights released with the preprint version of the PhaseNeXt paper.
They have been superseded. For current use, see `models02_v260828/`.**

## Why these weights were superseded

While performing a final check before publication, we found that waveform records for
certain station codes had been inadvertently omitted from the large-scale training data set
used for these weights. The omitted records account for approximately 2.7% of the intended
training data set. This was an omission of available data; no erroneous data were included.

We corrected the data set, retrained the models under the same conditions, and released the
resulting weights in `models02_v260828/`. The conclusions of the paper are unchanged, but
the numerical results differ slightly.

**These weights are retained so that any result obtained with the preprint version remains
reproducible.** Unless you are reproducing such a result, use `models02_v260828/`.

## Contents

| File | Description |
|---|---|
| `model_base.pt` | Weights of the base model (Model 2) |
| `model_PhaseNeXtS.pt` | Weights of PhaseNeXt-S (Model 10) |
| `model_PhaseNeXtM.pt` | Weights of PhaseNeXt-M (Model 19) |
| `model_base.py`, `model_PhaseNeXtS.py`, `model_PhaseNeXtM.py` | Model definitions |

**The model definitions (`.py`) are identical to those in `models02_v260828/`.**
Only the weights (`.pt`) differ. The network architectures were not changed.

## Checksums (MD5)

These files are frozen and will not change. Use these values to confirm which version you
have.

| File | MD5 |
|---|---|
| `model_base.pt` | `6fbe2e5823cbc15b9db781b95ae592a1` |
| `model_PhaseNeXtS.pt` | `ad66d7c76948bef235b16d1452224c58` |
| `model_PhaseNeXtM.pt` | `479a3bf15c7edd292c1d265006c93034` |

For reference, the corresponding files in `models02_v260828/` have the following checksums:

| File | MD5 |
|---|---|
| `model_base.pt` | `ca4828402accdc0e21780227c91b0ed7` |
| `model_PhaseNeXtS.pt` | `7043c5366a1394e02d2da2e01f9623b7` |
| `model_PhaseNeXtM.pt` | `e46c643740f953abcc1f63e0d0d13548` |

```
md5 model_base.pt model_PhaseNeXtS.pt model_PhaseNeXtM.pt      # macOS
md5sum model_base.pt model_PhaseNeXtS.pt model_PhaseNeXtM.pt   # Linux
```

## Usage

`run_PhaseNeXt.ipynb` in the repository root refers to `models02_v260828/`. To use the
weights in this directory instead, change the directory name in the notebook accordingly.
The model definitions are identical, so no other change is required.
