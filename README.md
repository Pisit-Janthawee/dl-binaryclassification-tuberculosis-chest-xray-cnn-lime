# Tuberculosis Chest X-ray: CNN + LIME

![Gradio screening app](assets/png/app_gradio.png)

Binary classification | Convolutional Neural Network (CNN) | EfficientNetB0 | Transfer learning | TensorFlow / Keras | Medical X-ray imaging | Contrast enhancement (CLAHE) | Near-duplicate detection | Grouped split | Class imbalance | LIME | Shortcut testing | Gradio

![What this project builds](assets/png/tb_overview.png)

## Objective

- **Problem** → tuberculosis still kills more than a million people a year; many clinics can take a chest X-ray but have no radiologist on duty, so X-rays wait
- **Why this repo** → train a CNN that gives every X-ray a probability of TB, set the threshold so it catches 90 % of TB cases, and test whether it looks at the lungs or at the hospital the X-ray came from
- **Real-world use** → a triage queue: likely-TB X-rays are read first and sent for a sputum test; the LIME map shows the doctor which regions drove the score
- **Business type** → healthcare, screening and triage
- **Learning problem** → binary image classification (normal 0 / TB 1)
- **Expected result** → P(TB) per X-ray + a screening decision + an explanation map

## Highlights

**Question:** is the model good enough to sort X-rays for a doctor?

**Answer:** on this dataset, yes. On 597 unseen X-rays it reaches ROC-AUC 0.997 and clears the WHO triage target (sensitivity ≥ 90 %, specificity ≥ 70 %) with room to spare.

| Test set (597 X-rays, 99 TB) | Value |
|---|---|
| ROC-AUC | **0.997** |
| Sensitivity (TB caught) | **93.9 %** · 93 of 99, 6 missed |
| Specificity (normal cleared) | **99.8 %** · 1 false alarm in 498 |
| Threshold (set on validation for 90 % sensitivity) | 0.881 |

![evaluation](assets/png/11_evaluation.png)

**Question:** is it reading the lungs, or the hospital?

**Answer:** partly the hospital. With the lungs blacked out, the same model still scores ROC-AUC 0.981. The sources explain why: every Belarus X-ray is TB, every Kaggle / RSNA X-ray is normal, and all 308 grayscale files are TB. The score is real on this dataset, but a new hospital's X-rays would likely score lower.

| Same model, test set | ROC-AUC | TB still flagged |
|---|---|---|
| Full image | 0.997 | 93.9 % |
| Lungs hidden | 0.981 | 49.5 % |
| Lungs only | 0.691 | 69.7 % |

![lungs hidden](assets/png/12_lungs_hidden.png)

**Question:** what does the average TB X-ray look like compared with a normal one?

**Answer:** brighter inside the lungs (denser tissue, the disease signal), but the largest differences sit at the top corners and borders (up to +60 grey levels): acquisition differences between sources.

![mean image](assets/png/05_mean_image.png)

**Question:** do TB lungs really look different, or only the rest of the X-ray?

**Answer:** both, but the source fingerprint is much stronger than the disease signal. Each X-ray gets one number per feature; the AUC is the chance that a TB X-ray has the higher value (0.5 = no difference). Every difference is statistically significant (Mann-Whitney, Bonferroni p < 0.004), so the AUC is what tells how big it is.

| Where | Number per X-ray | Normal → TB (median) | AUC | Size |
|---|---|---|---|---|
| file | colour channel difference (tinted files) | 0 → 9.8 | 0.78 | large |
| outside the lungs | contrast (std) | 66.7 → 53.7 | 0.28 | large |
| outside the lungs | black frame % | 17.4 → 8.3 | 0.34 | medium |
| lungs | upper-lung brightness | 84 → 101 | 0.63 | small |
| lungs | lung area % | 24.3 → 17.9 | 0.37 | small |
| lungs | lung brightness | 88 → 96 | 0.55 | negligible |

- **Inside the lungs** → TB lungs are brighter (denser tissue), most of all at the top where TB usually starts, and smaller; the grey-level curve moves from a peak near 90 to near 130
- **Outside and in the file** → only TB files have tinted colour channels, and TB X-rays have less contrast around the lungs → scanner and source, not disease

![normal vs TB](assets/png/08_normal_vs_tb.png)

## [1/8] Data

- **Dataset** → [Tuberculosis (TB) Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) · 3,500 normal + 700 TB · 512 × 512 PNG (not stored in this repository)
- **Imbalance** → 1 TB for every 5 normal → "always normal" is 83 % accurate and useless → ROC-AUC, sensitivity, specificity
- **Sources** → NLM Montgomery + Shenzhen (800, 49 % TB) · Belarus (306, all TB) · Kaggle / RSNA (3,094, all normal)
- **Files** → every normal file is RGB with R = G = B; the TB files are either grayscale (308) or RGB with **tinted** channels (392, photos of films) → the colour alone gives the class away; TB files are also smaller and more varied (median 142 kB vs 169 kB)

![samples](assets/png/04_samples.png)

## [2/8] Run on Kaggle

- **Import** → Create → Import Notebook → upload `tuberculosis_classification_cnn_lime.ipynb`
- **Input** → dataset `tawsifurrahman/tuberculosis-tb-chest-xray-dataset` (found anywhere under `/kaggle/input`)
- **Accelerator** → GPU (T4); Internet on (ImageNet weights, `pip install lime`)
- **Settings** → top cell; `SAVE_MODEL = False`, `LAUNCH_APP = False` → nothing is written, no app is started
- **App on Kaggle** → `LAUNCH_APP = True` prints a temporary public Gradio link

## [3/8] Near-duplicates and split

- **Difference hash** → 16 × 16 grid → 256-bit fingerprint; ≤ 10 differing bits = look-alike → 20 X-rays in 9 groups, none mixing classes
- **Why 10 of 256** → each bit asks "brighter than the right neighbour?", so a resized, brighter copy is only 2 bits away, while the median X-ray is 32 bits from its nearest neighbour
- **Split** → `StratifiedGroupKFold` by group → train 3,004 · validation 599 · test 597, each ~16.6 % TB

![pipeline](assets/png/tb_pipeline.png)

![difference hash](assets/png/m_dhash.png)

## [4/8] Image preparation

- **A picture is a table of numbers** → an RGB file holds 3 numbers per pixel (red, green, blue), a grayscale file 1 (0 = black air, 255 = white bone)
- **Grayscale 224 × 224** → every file is read as one grey channel ($0.299\,R + 0.587\,G + 0.114\,B$), which also removes the tint of the photographed films; copied to 3 channels per batch because the ImageNet backbone expects RGB

![from a PNG file to the model input](assets/png/m_image_array.png)

- **Why 224 and not the full 512?** → time and memory grow with the pixels, i.e. with the square of the side. A small test on a home CPU (same model shape, batches of 8):

| Image size | Pixels | One training step | All 4,200 X-rays in RAM |
|---|---|---|---|
| 64 × 64 | 4,096 (1×) | 0.11 s (1×) | 16 MB |
| 128 × 128 | 16,384 (4×) | 0.24 s (2.2×) | 66 MB |
| 256 × 256 | 65,536 (16×) | 0.97 s (8.7×) | 262 MB |

  Reading and resizing one PNG takes about 3.5 ms at every size; the model is where size costs time. 224 × 224 is also EfficientNetB0's pre-training size. `SIZE_BENCHMARK` in the notebook repeats the test (on Kaggle's GPU).

![image size vs time](assets/png/m_size_time.png)

- **CLAHE** → contrast per 8 × 8 tile, clip 2.0 → faint patches easier to see

![enhancement](assets/png/07_enhancement.png)

- **Lung mask** → Otsu threshold + morphology → rough, used only for the shortcut test

## [5/8] Model

- **EfficientNetB0** (ImageNet) + global average pooling + dropout 0.3 + sigmoid
- **Class weights** → normal 0.60, TB 2.99 (each TB X-ray counts ~5 ×)
- **Loss, part by part** → the sigmoid turns the raw score $z$ into P(TB) (the 0.881 threshold = $z > 2.0$); in the weighted binary cross-entropy the same doubt (P = 0.5) costs 2.07 on a TB X-ray and 0.42 on a normal one. The loss rewards the right label, not the right reason, so nothing in it stops a source shortcut
- **Online augmentation, clinically safe only** → rotation ±5°, zoom ±10 %, shift ±5 %, contrast ±10 %: changes a real X-ray could also have. Not used: left-right flip (heart on the right, like dextrocardia), upside-down flip, large rotations (lung tops leave the frame), shear and one-axis stretch (bent ribs, a different heart-to-chest ratio). Guidance: Elgendi et al., *Frontiers in Medicine*, 2021

![clinically safe and unsafe augmentation](assets/png/m_augment_clinical.png)
- **Training** → head (lr 1e-3, best val AUC 0.9938) → last 20 layers (lr 1e-5, best val AUC 0.9960)

![weighted binary cross-entropy](assets/png/m_bce_weighted.png)

![architecture](assets/png/tb_efficientnet_architecture.png)

![training curves](assets/png/10_training_curves.png)

## [6/8] Explanations (LIME)

- **LIME in one line** → hide parts of the X-ray and watch how P(TB) changes; the parts that change it most are the parts the CNN looks at
- **How** → cut the X-ray into ~60 pieces (superpixels) → 300 copies with about half the pieces painted black → P(TB) for every copy → a straight-line model gives each piece a weight
- **The straight line, part by part** → fitted with more trust in copies close to the original (half the pieces shown → weight 0.50, 10 % shown → 0.02) and a small ridge penalty; it assumes the pieces act one by one, and black holes are unlike any training X-ray, so a map is a check on one decision, not proof

![LIME step by step](assets/png/m_lime_2_perturbation.png)

- one test X-ray per confusion-matrix cell; green → towards TB, red → towards normal, cyan → lung mask
- true positive → green in the left lung, but also on a black border strip; misses → strongest regions on the abdomen and edges

![LIME](assets/png/13_lime.png)

## [7/8] App

- **Run locally** → `pip install -r requirements.txt` → `python app.py`
- **Model** → `models/tuberculosis_effnetb0.keras` + `models/settings.json` (threshold 0.881)
- **Examples** → set `TB_DATA_DIR` to the dataset folder; no X-rays are stored here
- **Output** → P(TB), a screening decision, a LIME map with the lung outline

## [8/8] Next steps

- **Source-held-out test** → train on some collections, test on another: the honest "new hospital" score
- **Lungs-only training** → U-Net lung segmentation (Montgomery / Shenzhen masks), classify the lungs only
- **Grad-CAM** → cross-check LIME with the CNN's own feature maps
- **More TB sources** → NIAID TB portal (data agreement required)
- **Calibration** → does P(TB) = 0.8 mean 8 in 10?

| Path | Content |
|---|---|
| `tuberculosis_classification_cnn_lime.ipynb` | full notebook (Kaggle-ready) |
| `app.py`, `requirements.txt` | Gradio screening app |
| `models/` | trained model and threshold |
| `assets/png/` · `assets/drawio/` | figures (PNG) · editable draw.io diagrams |
| `arch/` | the original notebook, kept for reference |
