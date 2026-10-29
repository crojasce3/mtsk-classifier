# LLM instructions

`deepseek_system_prompt.txt` is the exact system prompt used to classify the 278 documents with the DeepSeek API
(alias `deepseek-chat`, served as `deepseek-flash` on 29 September 2026; temperature 0.2; max. 512 output tokens).
It is reproduced verbatim, in Spanish, as used in the study.

Each document was sent as a user message with the format

```
TÍTULO: <title>
RESUMEN: <abstract>
PALABRAS CLAVE: <keywords or "No proporcionadas">
```

Note that the definitions in the prompt do not fully match those of the RedMTSK scheme used to label the corpus
(Table 1 of the article), and that the prompt includes a tie-breaking rule (T4 > T5 > T2 > T1 > T3). Both points are
discussed in the article.
