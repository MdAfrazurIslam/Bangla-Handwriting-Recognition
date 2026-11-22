"""Bengali handwriting recognition: drawing, photo upload, and camera."""
import json
import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "bangla_60_model.keras"
LABELS_PATH = HERE / "bangla_labels_60.json"

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    raise FileNotFoundError(
        "Keep bangla_60_model.keras and bangla_labels_60.json next to app.py"
    )

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with open(LABELS_PATH, encoding="utf-8") as f:
    labels = json.load(f)

assert len(labels) == 60 and model.output_shape[-1] == 60


def image_to_gray(photo):
    """Read a PIL image, with transparency flattened onto white."""
    photo = ImageOps.exif_transpose(photo)
    if "A" in photo.getbands() or "transparency" in photo.info:
        rgba = photo.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        photo = bg.convert("RGB")
    return np.asarray(photo.convert("L"), dtype=np.uint8)


def preprocess_photo(gray):
    """Remove uneven paper shading, isolate ink, and produce a 64x64 input."""
    gray = np.asarray(gray, dtype=np.uint8)
    h, w = gray.shape
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        gray = cv2.resize(
            gray, (round(w * scale), round(h * scale)),
            interpolation=cv2.INTER_AREA
        )
    h, w = gray.shape
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    if np.median(border) < 110:
        gray = 255 - gray

    sigma = max(3, min(h, w) * 0.08)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    contrast = cv2.subtract(background, gray)
    _, mask = cv2.threshold(
        contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    count, component_map, stats, centers = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    candidates = []
    for k in range(1, count):
        x, y, bw, bh, area = stats[k]
        if (
            area < max(8, int(gray.size * 0.0001))
            or x <= 1 or y <= 1
            or x + bw >= w - 1 or y + bh >= h - 1
        ):
            continue
        cx, cy = centers[k]
        distance = ((cx - w / 2) / (w / 2)) ** 2 + ((cy - h / 2) / (h / 2)) ** 2
        candidates.append((area / (1 + distance), k))

    if not candidates:
        raise ValueError("Could not find handwriting. Try a closer, clearer photo.")

    main = max(candidates)[1]
    x, y, bw, bh, main_area = stats[main]
    margin = int(0.8 * max(bw, bh))
    clean = np.zeros_like(mask)
    for _, k in candidates:
        area = stats[k, cv2.CC_STAT_AREA]
        cx, cy = centers[k]
        if (
            x - margin <= cx <= x + bw + margin
            and y - margin <= cy <= y + bh + margin
            and area >= max(5, main_area * 0.005)
        ):
            clean[component_map == k] = 255

    points = cv2.findNonZero(clean)
    if points is None:
        raise ValueError("Character extraction failed.")
    x, y, bw, bh = cv2.boundingRect(points)
    return resize_and_center(clean[y:y + bh, x:x + bw])


def resize_and_center(cropped):
    h, w = cropped.shape
    if h == 0 or w == 0:
        raise ValueError("No character found.")
    scale = min(56 / w, 56 / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=interpolation)
    canvas = np.zeros((64, 64), dtype=np.uint8)
    start_x, start_y = (64 - new_w) // 2, (64 - new_h) // 2
    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
    return canvas


def preprocess_drawing(editor_value):
    """Convert the Gradio sketchpad to the same grayscale CNN input size."""
    if editor_value is None:
        raise ValueError("Draw a character first.")
    image = (editor_value.get("composite")
             if isinstance(editor_value, dict) else editor_value)
    if image is None:
        raise ValueError("Draw a character first.")
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    gray = np.asarray(background.convert("L"), dtype=np.uint8)
    ink = 255 - gray
    ink[ink < 30] = 0
    if np.count_nonzero(ink) < 15:
        raise ValueError("Draw a visible Bengali character first.")
    points = cv2.findNonZero(ink)
    x, y, w, h = cv2.boundingRect(points)
    canvas = resize_and_center(ink[y:y + h, x:x + w])
    canvas = cv2.dilate(canvas, np.ones((2, 2), np.uint8), iterations=1)
    return canvas.astype(np.float32)[..., None] / 255.0


def predict_processed(image64):
    if image64.ndim == 2:
        image64 = image64.astype(np.float32)[..., None] / 255.0
    probabilities = model.predict(image64[None, ...], verbose=0)[0]
    index = int(np.argmax(probabilities))
    score = float(probabilities[index] * 100)
    result = (f"## {labels[index]}\n\n"
              f"**Class:** {index + 1}  |  **Model score:** {score:.2f}%")
    if score < 50:
        result += "\n\nLow-confidence prediction. Try another clear image."
    top = np.argsort(probabilities)[-3:][::-1]
    ranking = {f"{labels[i]} (Class {i + 1})": float(probabilities[i])
               for i in top}
    preview = Image.fromarray((image64[:, :, 0] * 255).astype(np.uint8))
    return result, preview, ranking


def recognize_photo(photo):
    if photo is None:
        return "Please upload or take a photo first.", None, {}
    try:
        gray = image_to_gray(photo)
        processed = preprocess_photo(gray)
        return predict_processed(processed)
    except ValueError as error:
        return str(error), None, {}


def recognize_drawing(sketch):
    try:
        return predict_processed(preprocess_drawing(sketch))
    except ValueError as error:
        return str(error), None, {}


# Prefer the outward-facing camera. Device/browser support varies.
rear_camera_options = gr.WebcamOptions(
    mirror=False,
    constraints={"facingMode": {"ideal": "environment"}}
)

with gr.Blocks(title="Bangla Handwriting Recognition") as demo:
    gr.Markdown("# বাংলা Handwriting Recognition")
    gr.Markdown("Draw a character, upload an image, or take a camera photo. "
                "Recognizes 60 Bengali letters, signs, and digits (not যুক্তবর্ণ).")

    with gr.Tab("✍️ Draw"):
        sketch = gr.Sketchpad(
            type="pil",
            label="Draw one Bengali character",
            canvas_size=(400, 400),
            brush=gr.Brush(
                default_size=18,
                colors=["#000000"],
                default_color="#000000",
                color_mode="fixed"
            ),
            layers=False,
            transforms=()
        )
        with gr.Row():
            recognize_draw_btn = gr.Button("Recognize Drawing", variant="primary")
            clear_btn = gr.Button("Clear")

    with gr.Tab("🖼️ Upload"):
        upload = gr.Image(sources=["upload"], type="pil", label="Upload handwriting")
        recognize_upload_btn = gr.Button("Recognize Upload", variant="primary")

    with gr.Tab("📷 Camera"):
        gr.Markdown("The app requests the rear camera where the browser supports it.")
        camera = gr.Image(
            sources=["webcam"], type="pil", label="Take a photo",
            webcam_options=rear_camera_options
        )
        recognize_camera_btn = gr.Button("Recognize Photo", variant="primary")

    gr.Markdown("## Recognition Result")
    result = gr.Markdown("Recognized character will appear here.")
    preview = gr.Image(label="CNN input (64×64)", interactive=False)
    top3 = gr.Label(label="Top 3 predictions", num_top_classes=3)

    outputs = [result, preview, top3]
    recognize_draw_btn.click(recognize_drawing, inputs=sketch, outputs=outputs)
    recognize_upload_btn.click(recognize_photo, inputs=upload, outputs=outputs)
    recognize_camera_btn.click(recognize_photo, inputs=camera, outputs=outputs)
    clear_btn.click(
        lambda: (None, "Draw a character to begin.", None, {}),
        outputs=[sketch, result, preview, top3]
    )

if __name__ == "__main__":
    # Public Colab link; hosting services provide their own URL.
    demo.launch(share=not bool(os.environ.get("SPACE_ID")))
