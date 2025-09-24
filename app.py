# app.py
import os, io, zipfile
from flask import Flask, request, send_file, jsonify
from pdf2image import convert_from_bytes
from PIL import Image
import numpy as np
import cv2
import pytesseract
import torch
from transformers import pipeline

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB limit

# Windows Tesseract path (uncomment and update if needed)
pytesseract.pytesseract.tesseract_cmd = r"F:\Tesseract-OCR\tesseract.exe"

# Models directory
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def get_sr(scale: int):
    """Return OpenCV DNN super-resolution object if model found in models/"""
    model_file = os.path.join(MODELS_DIR, f'EDSR_x{scale}.pb')
    if not os.path.exists(model_file):
        return None
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_file)
    sr.setModel('edsr', scale)
    return sr

# Device for transformers
device = 0 if torch.cuda.is_available() else -1

# Load smaller Hugging Face models
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", device=device)
sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
qa_pipe = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=device)
ner_pipe = pipeline("ner", grouped_entities=True, model="dbmdz/bert-large-cased-finetuned-conll03-english", device=device)

# --- Routes ---

@app.route("/")
def home():
    return """
    <h2>Welcome to PDF2JPG API</h2>
    <p>Available routes:</p>
    <ul>
        <li>POST /pdf-to-jpg</li>
        <li>POST /ocr</li>
        <li>POST /nlp/summarize</li>
        <li>POST /nlp/sentiment</li>
        <li>POST /nlp/keywords</li>
        <li>POST /nlp/qa</li>
    </ul>
    """

@app.route("/pdf-to-jpg", methods=["POST"])
def pdf_to_jpg():
    if 'file' not in request.files:
        return jsonify(error="file missing"), 400
    file = request.files['file']
    sr_flag = request.form.get('super_resolution', '0') == '1'
    scale = int(request.form.get('scale', '2'))

    try:
        pages = convert_from_bytes(file.read(), dpi=200)
    except Exception as e:
        return jsonify(error=str(e)), 500

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode='w') as zf:
        for i, page in enumerate(pages, start=1):
            img = page.convert('RGB')
            if sr_flag:
                sr = get_sr(scale)
                if sr:
                    np_img = np.array(img)
                    bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                    up = sr.upsample(bgr)
                    rgb_up = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb_up)
            out = io.BytesIO()
            img.save(out, format='JPEG', quality=90)
            out.seek(0)
            zf.writestr(f"page_{i}.jpg", out.read())

    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype="application/zip",
                     as_attachment=True, download_name="pages.zip")

@app.route("/ocr", methods=["POST"])
def ocr():
    if 'file' not in request.files:
        return jsonify(error="file missing"), 400
    file = request.files['file']
    fn = file.filename.lower()
    texts = []
    try:
        if fn.endswith('.pdf'):
            pages = convert_from_bytes(file.read(), dpi=200)
            for p in pages:
                texts.append(pytesseract.image_to_string(p))
        else:
            img = Image.open(file.stream).convert('RGB')
            texts.append(pytesseract.image_to_string(img))
    except Exception as e:
        return jsonify(error=str(e)), 500
    text = "\n".join([t for t in texts if t.strip()])
    return jsonify(text=text)

@app.route("/nlp/summarize", methods=["POST"])
def nlp_summarize():
    data = request.get_json(force=True)
    text = data.get('text', '')
    if not text.strip():
        return jsonify(error="text required"), 400
    out = summarizer(text, max_length=150, min_length=25, do_sample=False)
    return jsonify(summary=out[0]['summary_text'])

@app.route("/nlp/sentiment", methods=["POST"])
def nlp_sentiment():
    data = request.get_json(force=True)
    text = data.get('text', '')
    if not text:
        return jsonify(error="text required"), 400
    out = sentiment(text)
    return jsonify(out)

@app.route("/nlp/keywords", methods=["POST"])
def nlp_keywords():
    data = request.get_json(force=True)
    text = data.get('text', '')
    if not text:
        return jsonify(error="text required"), 400
    ents = ner_pipe(text)
    keywords = []
    for e in ents:
        w = e.get('word') or e.get('entity_group') or ''
        if w and w not in keywords:
            keywords.append(w)
    return jsonify(keywords=keywords)

@app.route("/nlp/qa", methods=["POST"])
def nlp_qa():
    data = request.get_json(force=True)
    context = data.get('context', '')
    question = data.get('question', '')
    if not context or not question:
        return jsonify(error="context and question required"), 400
    out = qa_pipe(question=question, context=context)
    return jsonify(out)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
