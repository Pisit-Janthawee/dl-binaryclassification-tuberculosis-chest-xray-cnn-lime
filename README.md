# Tuberculosis Chest X-ray: CNN + LIME

## [1/9] Overview

- **Data** → [Tuberculosis (TB) Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) · 3,500 normal + 700 TB X-rays · 512×512 PNG
- **Task** → chest X-ray → normal (0) / TB (1)
- **Metric** → ROC-AUC, plus sensitivity and specificity at a screening threshold (WHO triage target: ≥ 90% / ≥ 70%)
- **Notebook** → `tuberculosis_classification_cnn_lime.ipynb`
- **Backup** → `arch/` (original version)
- **Question** → is the model looking at the lungs, or at the hospital each X-ray came from?

## [2/9] Run on Kaggle

- **Import** → Create → Import Notebook → upload the `.ipynb`; the dataset is attached from the notebook metadata (otherwise Add Input → `tawsifurrahman/tuberculosis-tb-chest-xray-dataset`)
- **Accelerator** → GPU (T4)
- **Internet** → On (ImageNet weights, `pip install lime`)
- **Settings** → top cell of the notebook, documented in its note
- **Safety** → `SAVE_MODEL = False` → nothing is written
- **Lighter run** → lower `IMG_SIZE`, epochs or `LIME_SAMPLES`

## [3/9] What changed from the original version

- **Leakage fixed** → the original augmented and over-sampled TB images **before** the split → rotated twins of test images sat in training → its 0.99 accuracy was inflated
  - now → split first, near-duplicates grouped, augmentation only inside the model during training
- **Metric** → accuracy (83% for "always normal") → ROC-AUC, sensitivity, specificity
- **Threshold** → 0.5 → the highest threshold that still catches 90% of TB cases on validation
- **Model** → small CNN from scratch → EfficientNetB0 transfer learning, head then fine-tune
- **Shortcut check** → new: image sources, file properties, mean-image difference, lungs-hidden test

## [4/9] Data and shortcuts

- **Balance** → 700 TB of 4,200 (16.7%), 1 TB for every 5 normal
- **Sources** → metadata sheets give each image's URL → crosstab source × class
  - collections: NLM (Montgomery, Shenzhen), Belarus, NIAID TB portal, RSNA pneumonia challenge
  - RSNA has no TB labels → can only contribute normal X-rays → the class can be partly guessed from the source
- **File properties** → PNG mode (RGB / grayscale), size and file size per class, read from the headers only
- **Mean image** → average normal vs average TB X-ray and their difference map → differences inside the lungs = disease, at the borders = acquisition

## [5/9] Near-duplicates and split

- **Difference hash** → 16×16 grid → 256-bit fingerprint per X-ray, robust to resizing and small brightness changes
- **Hamming distance** → differing bits; ≤ `DUP_HAMMING` (10) → linked → linked images form one group
- **Split** → `StratifiedGroupKFold` → each group in exactly one split, similar TB share → ~70 / 15 / 15
  - the notebook stops with an assertion if a group appears in two splits

## [6/9] Image preparation

- **Loading** → one grayscale channel at `IMG_SIZE` (~200 MB at 224 px); copied to 3 channels per batch for the RGB backbone
- **Contrast enhancement** (`ENHANCE`) → contrast stretching, histogram equalization or CLAHE (default)
  - per image, nothing learned from other images → safe before the split
- **Lung mask** → CLAHE → Otsu threshold → clear border → two largest dark regions → closing, hole filling, dilation
  - rough by design → used for the shortcut test, not for diagnosis

## [7/9] Modeling

- **Imbalance** → class weights `n / (2 · n_class)` → TB ≈ 5× normal; no copies made
- **Online augmentation** → rotation ±5°, zoom ±10%, translation ±5%, contrast ±10%
  - **no flips** → a mirrored X-ray puts the heart on the right
- **Model** → EfficientNetB0 (ImageNet) + sigmoid head; frozen BatchNorm; mixed precision on GPU
- **[1/2] Head** → backbone frozen, Adam `1e-3`
- **[2/2] Fine-tune** → last `FINE_TUNE_LAST_N` layers, Adam `1e-5`; head weights restored if fine-tuning didn't beat them

## [8/9] Evaluation and where the model looks

- **Threshold** → set on validation for `TARGET_SENSITIVITY` = 0.90, reused unchanged on test
- **Charts** → ROC with the WHO target zone, precision-recall, probability histogram, confusion matrix
- **Lungs-hidden test** → the same model scores the test set three times: full image, lungs only, lungs hidden
  - lungs hidden near 0.5 → the model reads the lungs
  - lungs hidden still high → it uses cues outside the lungs (sources, markers, borders)
- **LIME** → one test image per confusion-matrix cell; green → towards TB, red → towards normal, cyan → lung mask

## [9/9] Next steps

- **Source-held-out test** → train on some collections, test on another
- **Learned lung segmentation** → U-Net on the Montgomery / Shenzhen lung masks, classify lungs only
- **Grad-CAM** → cross-check LIME with the CNN's own feature maps
- **More TB images** → 2,800 extra X-rays from the NIAID portal (data agreement required)
- **Calibration** → P(TB) = 0.8 should mean 8 in 10
