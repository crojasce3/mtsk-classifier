# Results files

| File | Content |
|---|---|
| `labels.csv` | One row per document: `row` (position in the curated corpus, 0-based), `original_id` (row in the original 293-record collection), `label` (thematic area T1-T5, final version after curation), `language` (es/en/pt), `related_versions_group` (shared code for related versions of the same study), `cv_fold` (0-4). |
| `cross_validation/folds.json` | Training, internal validation and test indices of each fold. |
| `cross_validation/oof_probabilities/*.npy` | Out-of-fold class probabilities (278 × 5, columns T1-T5, rows in corpus order) for `e5-large`, `e5-large-256`, `minilm`, `xlmr-base`, `xlmr-base-2e5` and `tfidf-lr`. |
| `cross_validation/fold_metrics/*.csv` | Per-fold test metrics as recorded during training. |
| `final_model/test_predictions_fold0.csv` | Predictions and class probabilities of the fold-0 model on its 56 test documents (`doc` is the index used in the SHAP analysis and in the article's qualitative examples). |
| `final_model/shap_word_level_by_class.csv` | Word-level SHAP support for each area (Figure 2). |
| `final_model/training_history.csv`, `test_summary.json`, `test_per_class.csv` | Training curve and test metrics of the fold-0 model. |
| `llm/deepseek_predictions.csv` | LLM label, secondary areas, full response, API model name, query date and token counts for each document. |
| `expert/expert_annotation.csv` | Independent expert's area, confidence (1-3), optional second area and justification for the 100 sampled documents (anonymised). |
