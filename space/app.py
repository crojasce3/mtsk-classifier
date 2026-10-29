"""
Clasificador MTSK: clasificación explicable de la investigación fundamentada en el
modelo MTSK según las cinco áreas temáticas de la RedMTSK.

Modelo: intfloat/multilingual-e5-large afinado (pliegue 0 de la validación cruzada).
Entrada: "query: " + título + resumen + palabras clave, truncada a 128 tokens.
"""
import os
import re
import tempfile

# `spaces` must be imported before torch so that the Space also runs on ZeroGPU hardware.
try:
    import spaces
    GPU = spaces.GPU
except ImportError:          # running locally or on CPU without the `spaces` package
    def GPU(fn=None, **kwargs):
        return fn if fn is not None else (lambda f: f)

import gradio as gr
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

MODEL_REPO = os.environ.get("MODEL_REPO", "crojasc2/mtsk-classifier")
REVIEW_THRESHOLD = 0.885   # mediana de los umbrales seleccionados de forma anidada (0.855–0.905)
MAX_BATCH_ROWS = 500

AREAS = {
    "T1": "MTSK en la formación de profesores",
    "T2": "El formador de profesores de matemáticas",
    "T3": "MTSK en temas y etapas específicos",
    "T4": "Desarrollo teórico del modelo",
    "T5": "Extensiones del modelo",
}

# ── Carga del modelo ─────────────────────────────────────────────────────────
torch.set_num_threads(max(1, os.cpu_count() or 1))
ZERO_GPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "true")
DEVICE = torch.device("cuda" if ZERO_GPU or torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
backbone = AutoModel.from_pretrained(MODEL_REPO)
head_ckpt = torch.load(hf_hub_download(MODEL_REPO, "classifier_head.pt"), map_location="cpu")

LABELS = head_ckpt.get("nam_labels", list(AREAS))
PREFIX = head_ckpt.get("prefix", "query: ")
MAX_LEN = int(head_ckpt.get("max_length", 128))
DROPOUT = float(head_ckpt.get("dropout", 0.3))
head_state = head_ckpt.get("classifier_state_dict", head_ckpt.get("classification_head"))


class MTSKClassifier(torch.nn.Module):
    """[CLS] (primer token) → Dropout → Linear, igual que en el entrenamiento."""

    def __init__(self, backbone, num_labels, p=DROPOUT):
        super().__init__()
        self.backbone = backbone
        self.dropout = torch.nn.Dropout(p)
        self.classifier = torch.nn.Linear(backbone.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(h))


model = MTSKClassifier(backbone, len(LABELS))
model.classifier.load_state_dict(head_state)
model.to(DEVICE).eval()


def build_text(title, abstract, keywords):
    parts = [str(x).strip() for x in (title, abstract, keywords) if x is not None and str(x).strip() and str(x) != "nan"]
    return (PREFIX + " ".join(parts)).strip()


@torch.no_grad()
def predict_proba(texts, batch_size=16):
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(list(texts[i:i + batch_size]), truncation=True, padding=True,
                        max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        logits = model(enc["input_ids"], enc["attention_mask"])
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(LABELS)))


# ── Explicación con SHAP (opcional, más lenta) ───────────────────────────────
def explain(text, target_idx, max_evals=300):
    import shap

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(lambda x: predict_proba([str(t) for t in x]), masker, output_names=LABELS)
    sv = explainer([text], max_evals=max_evals)[0]
    # agregación a nivel de palabra, como en el artículo
    words, cur, val = [], "", 0.0
    for tok, v in zip(sv.data, sv.values[:, target_idx]):
        tok = str(tok)
        if not tok:
            continue
        cur += tok
        val += float(v)
        if tok.endswith(" ") or tok.endswith("\n"):
            words.append((cur, val))
            cur, val = "", 0.0
    if cur:
        words.append((cur, val))
    # quitar el prefijo "query: "
    while words and words[0][0].strip().lower().rstrip(":") in ("query", "que", "ry"):
        words.pop(0)
    scale = max((abs(v) for _, v in words), default=1.0) or 1.0
    return [(w, round(v / scale, 3)) for w, v in words]


# ── Clasificación de un documento ────────────────────────────────────────────
@GPU(duration=120)
def classify(title, abstract, keywords, do_explain):
    if not (title or abstract or keywords):
        raise gr.Error("Escriba al menos el título del documento.")
    text = build_text(title, abstract, keywords)
    probs = predict_proba([text])[0]
    k = int(probs.argmax())
    conf = float(probs[k])
    label_probs = {f"{l} · {AREAS.get(l, '')}": float(p) for l, p in zip(LABELS, probs)}
    n_tokens = len(tokenizer(text)["input_ids"])
    msg = [f"**Área predicha:** {LABELS[k]} · {AREAS.get(LABELS[k], '')}",
           f"**Confianza:** {conf:.3f}"]
    if conf < REVIEW_THRESHOLD:
        msg.append(f"⚠️ La confianza es inferior a {REVIEW_THRESHOLD}. Se recomienda revisión experta.")
    else:
        msg.append("✅ Confianza alta. Aun así, el modelo puede equivocarse con alta confianza, "
                   "sobre todo entre T3 y T4 y en documentos de T5.")
    if n_tokens > MAX_LEN:
        msg.append(f"ℹ️ El texto tiene {n_tokens} tokens; el modelo solo lee los primeros {MAX_LEN}.")
    highlights = explain(text, k) if do_explain else None
    return label_probs, "\n\n".join(msg), highlights


# ── Clasificación por lotes ──────────────────────────────────────────────────
COLS = {
    "title": ["título", "titulo", "title"],
    "abstract": ["resumen", "resumen (texto literal)", "abstract"],
    "keywords": ["palabras clave", "palabras clave (texto literal)", "keywords"],
}


def _find(df, names):
    low = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    return None


@GPU(duration=120)
def classify_file(file):
    if file is None:
        raise gr.Error("Suba un archivo .xlsx o .csv.")
    path = file if isinstance(file, str) else file.name
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    if len(df) > MAX_BATCH_ROWS:
        raise gr.Error(f"El archivo tiene {len(df)} filas; el máximo es {MAX_BATCH_ROWS}.")
    cols = {k: _find(df, v) for k, v in COLS.items()}
    if cols["title"] is None and cols["abstract"] is None:
        raise gr.Error("No se encontraron columnas de título ni de resumen (Título, Resumen, Palabras clave).")
    get = lambda c: df[c] if c is not None else pd.Series([""] * len(df))
    texts = [build_text(t, a, k) for t, a, k in zip(get(cols["title"]), get(cols["abstract"]), get(cols["keywords"]))]
    probs = predict_proba(texts)
    out = df.copy()
    out["Área predicha"] = [LABELS[i] for i in probs.argmax(1)]
    out["Nombre del área"] = [AREAS.get(LABELS[i], "") for i in probs.argmax(1)]
    out["Confianza"] = probs.max(1).round(4)
    out["Revisión sugerida"] = np.where(probs.max(1) < REVIEW_THRESHOLD, "Sí", "No")
    for j, l in enumerate(LABELS):
        out[f"P({l})"] = probs[:, j].round(4)
    tmp = os.path.join(tempfile.mkdtemp(), "clasificacion_MTSK.xlsx")
    out.to_excel(tmp, index=False)
    summary = out["Área predicha"].value_counts().reindex(LABELS, fill_value=0)
    resumen = (f"{len(out)} documentos clasificados. "
               + ", ".join(f"{l}: {n}" for l, n in summary.items())
               + f". Revisión sugerida en {int((out['Revisión sugerida'] == 'Sí').sum())} documentos.")
    return tmp, resumen


# ── Interfaz ─────────────────────────────────────────────────────────────────
DESCRIPTION = f"""
Clasifica trabajos de investigación fundamentados en el modelo **MTSK** (*Mathematics Teacher's Specialised Knowledge*)
en las cinco áreas temáticas de la RedMTSK, a partir de su título, resumen y palabras clave, en español, inglés o portugués.

{chr(10).join(f"- **{k}** · {v}" for k, v in AREAS.items())}

El modelo es un apoyo a la revisión experta, no un sustituto. Las predicciones con confianza inferior a {REVIEW_THRESHOLD}
deberían revisarse.
"""

with gr.Blocks(title="Clasificador MTSK") as demo:
    gr.Markdown("# Clasificador MTSK")
    gr.Markdown(DESCRIPTION)
    with gr.Tab("Un documento"):
        with gr.Row():
            with gr.Column():
                t_in = gr.Textbox(label="Título", lines=2)
                a_in = gr.Textbox(label="Resumen", lines=8)
                k_in = gr.Textbox(label="Palabras clave", lines=1)
                x_in = gr.Checkbox(label="Mostrar explicación con SHAP (tarda alrededor de un minuto)", value=False)
                btn = gr.Button("Clasificar", variant="primary")
            with gr.Column():
                probs_out = gr.Label(label="Probabilidad por área", num_top_classes=5)
                msg_out = gr.Markdown()
                expl_out = gr.HighlightedText(
                    label="Contribución de cada palabra al área predicha (rojo: a favor; azul: en contra)",
                    combine_adjacent=False, show_legend=False,
                    color_map=None)
        btn.click(classify, [t_in, a_in, k_in, x_in], [probs_out, msg_out, expl_out])
    with gr.Tab("Varios documentos"):
        gr.Markdown("Suba un archivo **.xlsx** o **.csv** con columnas *Título*, *Resumen* y *Palabras clave* "
                    f"(también se aceptan *Title*, *Abstract*, *Keywords*). Máximo {MAX_BATCH_ROWS} filas.")
        f_in = gr.File(label="Archivo", file_types=[".xlsx", ".csv"])
        f_btn = gr.Button("Clasificar archivo", variant="primary")
        f_out = gr.File(label="Resultado")
        s_out = gr.Markdown()
        f_btn.click(classify_file, [f_in], [f_out, s_out])
    gr.Markdown("Rojas Celis, C., Elorreaga Rodríguez, L. M. y Hortúa, H. J. Universidad El Bosque. "
                "Código: https://github.com/crojasce3/mtsk-classifier")

if __name__ == "__main__":
    demo.launch()
