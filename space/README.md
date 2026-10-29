---
title: Clasificador MTSK
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Clasificador MTSK

Clasificación explicable de la investigación fundamentada en el modelo MTSK (*Mathematics Teacher's Specialised Knowledge*) según las cinco áreas temáticas de la RedMTSK, a partir del título, el resumen y las palabras clave de cada documento, en español, inglés o portugués.

- **Modelo:** `intfloat/multilingual-e5-large` afinado sobre un corpus de 278 documentos etiquetados por expertas (pesos en [crojasc2/mtsk-classifier](https://huggingface.co/crojasc2/mtsk-classifier)).
- **Desempeño en validación cruzada estratificada de 5 pliegues:** macro-F1 = 0.732 (IC 95 %, 0.677–0.782).
- **Explicabilidad:** atribuciones SHAP agregadas a nivel de palabra (opcional en la interfaz).
- **Uso recomendado:** como apoyo a la revisión experta. Las predicciones con confianza inferior a 0.885 deberían revisarse; el modelo puede equivocarse con alta confianza, sobre todo entre T3 y T4 y en documentos de T5.

Código: https://github.com/crojasce3/mtsk-classifier
