# Hybrid Foundation Model (HFM) for Industrial Visual Inspection

Code for the paper **"Industrial Visual Inspection Using Self-Supervised, Few-Shot and Zero-Shot Learning"** (Hedieh Sajedi, Abolfazl Khojasteh Abkenar), revised version.


## What this repository contains

HFM uses two frozen foundation models and no training:

- **P**: a prototype branch (DINOv2 ViT-L/14 features, class prototypes).
- **A**: a PatchCore-style memory-bank branch (DINOv2 patch features, random 15% bank, at most 6,000 patches, top-3 mean score). It also produces the anomaly maps.
- **Z**: a zero-shot text branch (CLIP ViT-L/14 with text prompts).
- Scores are combined by a training-free variance-softmax fusion. Other fusion rules are included for comparison.

**Main finding of the revised paper:** the memory-bank branch alone detects defects better than the full three-branch model on five of the six datasets. The variance weights are close to uniform. The extra branches add capabilities (few-shot classification, text-based recognition), not detection accuracy. Please read Sections 5.5 and 6 of the paper.

| File | Purpose |
|---|---|
| `hfm_script1_mvtec_visa_btad_mpdd.py` | MVTec AD, VisA, BTAD, MPDD: all main experiments |
| `output_script1.zip` | Output of hfm_script1 |
| `hfm_script2_mvtec_ad2_and_crack_domain.py` | MVTec AD 2 and the concrete-crack dataset |
| `output_script2.zip` | Output of hfm_script2 |
| `README.md` | This file |

## Evaluation protocol (no test leakage)

For every category and every seed:

1. Normal training images (at most 500 per category) are split 80% **reference** / 20% **validation**.
2. The memory bank and prototypes come from the reference set only.
3. **Calibration regime:** 20% of the test defects (at least 2) form a calibration set and are removed from evaluation. Min–max constants, fusion weights and the F1-maximising threshold are fitted on validation normals plus calibration defects. All metrics are measured on the remaining evaluation images.
4. **Normal-only regime:** no defect label is used. Scores are z-scored with validation normals, the threshold is the 95th or 99th percentile of validation-normal fused scores, and the result is measured on the full test set.
5. Values tuned on test labels appear only as clearly named *oracle* upper bounds.
6. Ground-truth masks get the same resize and centre crop as the images (nearest neighbour), so masks and anomaly maps are aligned.

Ten seeds are used (`N_SEEDS = 10`, `SEED_BASE = 1000`). Localization, robustness and bank-size studies use three seeds.

## How to run (Google Colab, T4 GPU)

1. Open a new Colab notebook. Choose **Runtime > Change runtime type > T4 GPU**.
2. Upload a script, then run it with `%run` (not `!python`, because the Kaggle upload dialog and the final download need the notebook):
   ```python
   %run hfm_script1_mvtec_visa_btad_mpdd.py
   ```
3. The first lines of the script install every library it needs.
4. When it asks, upload your `kaggle.json` (Kaggle: Settings > Create New API Token). Alternatively set `KAGGLE_USERNAME` and `KAGGLE_KEY`.
5. Run the second script the same way, preferably in a fresh runtime:
   ```python
   %run hfm_script2_mvtec_ad2_and_crack_domain.py
   ```
6. Each script writes `output_script1.zip` or `output_script2.zip` to `/content` and starts the download.

Approximate run time on a T4: about 135 minutes (script 1) and about 62 minutes (script 2).

**Resuming.** Results are cached in `/content/hfm_cache/results`. If the session stops, run the same script again and finished work is skipped, as long as the runtime was not recycled. Set `USE_DRIVE_BACKUP = True` in the script to copy checkpoints and the zip to Google Drive.

**Smoke test.** `HFM_QUICK=1` runs a very small version to check that everything works:
```python
import os; os.environ["HFM_QUICK"] = "1"
%run hfm_script1_mvtec_visa_btad_mpdd.py
```
Do not use quick-mode numbers in any report.

## Datasets

Downloaded automatically (Kaggle credentials needed unless noted):

| Dataset | Source |
|---|---|
| MVTec AD | Kaggle `ipythonx/mvtec-ad` |
| VisA | official Amazon archive (downloaded by script 1, no Kaggle needed) |
| BTAD | Kaggle `thtuan/btad-beantech-anomaly-detection` |
| MPDD | Kaggle `lephonghao/metal-parts-defect-detection-dataset-mpdd` |
| MVTec AD 2 | official archive (public test set), script 2 |
| Concrete cracks | Kaggle `arunrk7/surface-crack-detection` and `thesenimark/concrete-crack-images-for-classification` (no pixel masks) |

Please check each dataset's own licence before use. The datasets are not redistributed here.

## Output of each script

The zip contains `tables/` (CSV), `figures/` (PNG), `raw/` (per-seed results, `.csv.gz`), `code/` (copy of the script), `environment.json`, `requirements_frozen.txt`, `run_log.txt` and `RESULTS_SUMMARY.md`.

## Where results appear in the paper

| Paper item | Content |
|---|---|
| Table 4, Figure 7 | Main image-level results, confusion matrices |
| Table 5, Table 6 | Few-shot prototype; normal-only versus normal+defect |
| Table 7 | Zero-shot recognition, chance and majority-class baselines |
| Table 8, Table 9 | Anomaly detection; comparison with published numbers |
| Table 10, Figure 8 | Localization (pixel AUROC/AP, AU-PRO, Dice, boundary F, region metrics) |
| Table 11, Table 12, Figure 10 | Ablation by removal; fusion rules |
| Table 13, Figure 11 | Paired statistics; variance as a reliability signal |
| Table 14 | Normal-only regime |
| Table 15, Figure 13 | Parameters, MACs, latency by stage, memory |
| Table 16, Figure 12 | Robustness to image changes |
| Table 17 | Failure analysis |

## Environment

Frozen versions are in `requirements_frozen.txt` inside each output zip. The reported run used Python on Colab with `torch 2.11.0+cu130` and `transformers 5.20.2`. Models are `facebook/dinov2-large` and `openai/clip-vit-large-patch14`, downloaded from Hugging Face.

## Limits you should know

- Zero-shot prompts are weak; knowing defect names does not improve detection.
- Few-shot defect supports are taken from the test split and removed from the queries.
- Compute cost is high (about 121 ms per image on a T4, 608 M parameters at inference).
- Comparison with other methods uses their published numbers, not re-runs.
