# models02_v260828 — current version

**These are the current model weights for Naoi et al. (under review in Earth, Planets and
Space).** Use these unless you are reproducing a result obtained with the preprint version.

Released: 2026-08-28. Supersedes the weights in `models02_preprint_v260416/`.

## Contents

| File | Description |
|---|---|
| `model_base.pt` | Weights of the base model (Model 2) |
| `model_PhaseNeXtS.pt` | Weights of PhaseNeXt-S (Model 10) |
| `model_PhaseNeXtM.pt` | Weights of PhaseNeXt-M (Model 19) |
| `model_base.py`, `model_PhaseNeXtS.py`, `model_PhaseNeXtM.py` | Model definitions |

## Notes on use

- The models assume **fixed-length waveforms of 4096 samples**.
- Waveform input is ordered in **UNE** channel order; output is returned in **PSN** order.
- The **final softmax layer is not included** in these models, following standard PyTorch
  conventions. (The models distributed through SeisBench do include it.)
- Only events labeled as ordinary earthquakes in the JMA Unified Catalog were used for
  training.
- `run_PhaseNeXt.ipynb` in the repository root is a sample program for these models.

## What changed from the preprint version

While performing a final check before publication, we found that waveform records for
certain station codes had been inadvertently omitted from the large-scale training data set
used for the earlier weights. The omitted records account for approximately 2.7% of the
intended training data set. This was an omission of available data; no erroneous data were
included.

We corrected the data set and retrained the models under the same conditions. The
conclusions of the paper are unchanged, but the numerical results differ slightly.

**The model definitions (`.py`) are identical to those in `models02_preprint_v260416/`.**
Only the weights (`.pt`) differ. The network architectures were not changed.

## Checksums (MD5)

| File | MD5 |
|---|---|
| `model_base.pt` | `ca4828402accdc0e21780227c91b0ed7` |
| `model_PhaseNeXtS.pt` | `7043c5366a1394e02d2da2e01f9623b7` |
| `model_PhaseNeXtM.pt` | `e46c643740f953abcc1f63e0d0d13548` |

For reference, the superseded weights in `models02_preprint_v260416/`:

| File | MD5 |
|---|---|
| `model_base.pt` | `6fbe2e5823cbc15b9db781b95ae592a1` |
| `model_PhaseNeXtS.pt` | `ad66d7c76948bef235b16d1452224c58` |
| `model_PhaseNeXtM.pt` | `479a3bf15c7edd292c1d265006c93034` |

```
md5 model_base.pt model_PhaseNeXtS.pt model_PhaseNeXtM.pt      # macOS
md5sum model_base.pt model_PhaseNeXtS.pt model_PhaseNeXtM.pt   # Linux
```
