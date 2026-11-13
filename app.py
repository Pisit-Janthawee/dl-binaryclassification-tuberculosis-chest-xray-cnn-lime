"""Tuberculosis chest X-ray screening demo (Gradio).

Upload a chest X-ray → the fine-tuned EfficientNetB0 returns P(TB), a screening decision and a LIME map.
Preprocessing matches the notebook: grayscale, 224 x 224 (INTER_AREA), CLAHE (clip 2.0, 8 x 8 tiles).

    pip install -r requirements.txt
    python app.py

Optional environment variables
    TB_MODEL      path to the .keras model    (default models/tuberculosis_effnetb0.keras)
    TB_DATA_DIR   dataset folder with Normal/ and Tuberculosis/ → example images in the app (none are stored here)
    TB_DEMO       1 → run the first example on page load (used for the README screenshot)
"""
import json
import os
from pathlib import Path

import cv2
import gradio as gr
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border, mark_boundaries

HERE = Path(__file__).resolve().parent
MODEL_FILE = Path(os.environ.get('TB_MODEL', HERE / 'models' / 'tuberculosis_effnetb0.keras'))
SETTINGS = json.loads((HERE / 'models' / 'settings.json').read_text()) if (HERE / 'models' / 'settings.json').exists() else {}
IMG_SIZE = SETTINGS.get('img_size', 224)
THRESHOLD = SETTINGS.get('threshold', 0.5)
LIME_SAMPLES = int(os.environ.get('TB_LIME_SAMPLES', 200))

import keras  # noqa: E402  (after matplotlib backend)

MODEL = keras.models.load_model(MODEL_FILE, compile=False)
EXPLAINER = lime_image.LimeImageExplainer(random_state=42)
SEGMENTER = SegmentationAlgorithm('slic', n_segments=60, compactness=10, start_label=0, random_seed=42)


# ---------------------------------------------------------------- preprocessing (same as the notebook)
def clahe(image):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)


def prepare(image):
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return clahe(cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA))


def lung_mask(image):
    size = image.shape[0]
    kernel = lambda f: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(round(size * f)) | 1),) * 2)
    _, dark = cv2.threshold(cv2.GaussianBlur(clahe(image), (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    labels = label(clear_border(dark > 0))
    largest = sorted(regionprops(labels), key=lambda r: r.area, reverse=True)[:2]
    lungs = np.isin(labels, [r.label for r in largest]).astype(np.uint8)
    filled = ndimage.binary_fill_holes(cv2.morphologyEx(lungs, cv2.MORPH_CLOSE, kernel(0.07))).astype(np.uint8)
    return cv2.dilate(filled, kernel(0.03)).astype(bool)


def predict_rgb(images):
    p = MODEL.predict(np.asarray(images, dtype=np.float32), batch_size=32, verbose=0).ravel()
    return np.stack([1 - p, p], axis=1)


# ---------------------------------------------------------------- app logic
def screen(image):
    if image is None:
        return None, 'Upload a chest X-ray.', None
    x = prepare(np.asarray(image))
    rgb = np.repeat(x[..., None], 3, axis=-1)
    p = float(predict_rgb(rgb[None])[0, 1])
    flagged = p >= THRESHOLD
    decision = (f'### {"Possible TB → refer for a sputum test" if flagged else "No TB pattern found → routine care"}\n'
                f'P(TB) = **{p:.3f}** · screening threshold {THRESHOLD:.3f} (set to catch 90 % of TB cases on validation)\n\n'
                f'A screening aid that sorts X-rays for a doctor, not a diagnosis.')
    explanation = EXPLAINER.explain_instance(rgb, predict_rgb, labels=(1,), hide_color=0, segmentation_fn=SEGMENTER,
                                             num_samples=LIME_SAMPLES, batch_size=32, random_seed=42)
    img, mask = explanation.get_image_and_mask(1, positive_only=False, num_features=10, hide_rest=False)
    lungs = lung_mask(x)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    axes[0].imshow(x, cmap='gray')
    axes[0].set_title('Input after CLAHE (cyan = lung mask)')
    axes[1].imshow(mark_boundaries(img / 255.0, mask))
    axes[1].set_title('LIME: green → towards TB, red → towards normal')
    for ax in axes:
        ax.contour(lungs, colors='cyan', linewidths=1.2)
        ax.axis('off')
    fig.tight_layout()
    return {'Tuberculosis': p, 'Normal': 1 - p}, decision, fig


def examples():
    root = os.environ.get('TB_DATA_DIR')
    if not root:
        return []
    root = Path(root)
    found = []
    for cls in ('Tuberculosis', 'Normal'):
        folder = next((p for p in root.rglob(cls) if p.is_dir()), None)
        if folder:
            found += sorted(folder.glob('*.png'))[:4]
    return [[str(p)] for p in found]


EXAMPLES = examples()
with gr.Blocks(title='Tuberculosis X-ray screening') as demo:
    gr.Markdown('# Tuberculosis Chest X-ray Screening\n'
                'Upload a chest X-ray. The fine-tuned EfficientNetB0 returns the probability of tuberculosis, '
                'a screening decision and a LIME map of the regions behind the score.')
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type='numpy', label='Chest X-ray', height=380)
            btn = gr.Button('Screen', variant='primary')
            if EXAMPLES:
                gr.Examples(EXAMPLES, inputs=inp)
        with gr.Column():
            prob = gr.Label(num_top_classes=2, label='Probability')
            text = gr.Markdown()
            plot = gr.Plot(label='Explanation', show_label=False)
    btn.click(screen, inputs=inp, outputs=[prob, text, plot])
    if os.environ.get('TB_DEMO') == '1' and EXAMPLES:
        first = cv2.cvtColor(cv2.imread(EXAMPLES[0][0]), cv2.COLOR_BGR2RGB)
        cached = (first, *screen(first))                 # computed once at start-up, shown on every page load
        demo.load(lambda: cached, outputs=[inp, prob, text, plot])

if __name__ == '__main__':
    demo.launch()
