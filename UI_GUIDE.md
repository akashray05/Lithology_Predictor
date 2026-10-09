# LITHO-ML Web Interface — User Guide

LITHO-ML predicts lithology from well-log data using three frozen models trained on the FORCE 2020
dataset: **Ex-Tree** (ExtraTrees, internal name W5), **XGBoost** and **LightGBM**. The web interface lets you
upload a well, run all three models, compare them visually and numerically, and export the results.

> Predictions are estimates, not verified geological truth. Metrics are only calculated when the uploaded
> file contains original lithology labels.

---

## 1. Run the app

```bash
cd Lithology_Predictor
source .venv/bin/activate
streamlit run app_v2.py
```

The app opens at `http://localhost:8501`. To verify the installation first:

```bash
python tests/verify_ui.py
python tests/verify_ui.py "data/splits/wells/blind_test/31_2-10.las"
```

## 2. Input file

| Requirement | Detail |
|---|---|
| Format | `.las` or `.csv` |
| Required curves | `GR`, `RDEP`, `RMED`, `DTC`, `RHOB` |
| Depth | `DEPT` or `DEPTH` column (if missing, the sample index is used) |
| Optional labels | A lithology column with FORCE codes (e.g. `65000`) or names (e.g. `Shale`). Auto-detected from `FORCE_2020_LITHOFACIES_LITHOLOGY`, `LITHOLOGY`, `TRUE_LITHOLOGY`, `LITHOFACIES`, `FACIES` or `LABEL`. |

Before prediction, invalid values are set to missing (GR < 0; DTC, RHOB, RDEP, RMED ≤ 0), resistivities are
log10-transformed, and 41 features are built per well (5 base logs, 30 rolling statistics over windows of
5/9/21 samples, 1 resistivity separation, 5 missingness indicators). No values are imputed.

## 3. Sidebar

**Navigate** switches between the **Predictor** and the **Model guide**. Your uploaded file is kept while
you read the guide.

| Section | What it controls |
|---|---|
| ① Models | Which of Ex-Tree, XGBoost and LightGBM are plotted, compared and exported. |
| ② Original lithology | Which column holds the reference lithology (auto-detect, none, or choose one). |
| ③ View | Depth interval, original-lithology track, error strips, labelled-interval limit, confidence track, log curves, plot height. |

## 4. Tabs

### 🧭 Overview
Sample count, depth interval, number of labelled samples, and the share of samples where all selected models
agree. Below: lithology distribution (original vs each model), a model summary (classes predicted, mean
confidence, accuracy when labels exist) and a table of the thickest depth intervals where the models disagree.

### 📊 Tracks
Depth-aligned lithology columns: selected log curves, the original lithology, and each model's prediction
with a green/red error strip (green = matches the original, red = differs, blank = no label). Drag to zoom,
double-click to reset, scroll to zoom the depth axis. The camera icon saves a PNG.

### 🔀 Compare
Model-versus-model views that need no labels: pairwise agreement between models, a confidence histogram, and
a consensus (majority-vote) track with an agreement strip. **Agreement is not accuracy** — models can agree and
all be wrong.

### ✅ Evaluate
Needs original labels. Results are for the uploaded well only.

- **Metrics** — accuracy, balanced accuracy, macro F1, weighted F1 and the official FORCE penalty score.
- **Confusion & classes** — confusion matrix (percent of true class or counts) and per-class precision, recall, F1.
- **Where errors happen** — rolling accuracy against depth, and accuracy near true lithology boundaries versus inside beds.
- **Calibration** — observed accuracy against model confidence. Points below the dashed line mean over-confidence.

### 🧪 Geology
Checks whether predictions make petrophysical sense.

- **Original vs model(s)** — side-by-side cross-plots (GR–RHOB, DTC–RHOB, RDEP–GR, RDEP–RMED) using the same
  samples and axes, so colour differences are real differences in lithology. Each model panel shows how often it
  agrees with the original. Tick **Show wrong predictions** to ring only the points where the model disagrees.
- **Single plot** — one cross-plot coloured by the original or any model.
- **Max points** is set in multiples of 500 for responsiveness.
- **Bed thickness** — number and thickness of contiguous beds per lithology. Many thin beds in a prediction
  compared with the original suggests salt-and-pepper noise.

### 💾 Export
Preview of the uploaded data and the prediction table, with downloads for predictions (CSV, including
consensus and agreement columns) and metrics (CSV, only when labels exist).

## 5. How to read the results honestly

1. **Use wells the models have not seen.** Wells in `data/splits/wells/train/` were used for training, so
   their scores are optimistic. For an honest check, upload a well from `data/splits/wells/blind_test/`.
2. **One well is not generalisation.** The Evaluate tab describes a single well. Overall performance is the
   10-well blind-test result shown in the Model guide.
3. **No labels, no accuracy.** For an unlabelled well the app shows predictions and confidence only.
4. **Confidence is not certainty.** It is the probability assigned to the predicted class and may not be
   calibrated; use the Calibration view to check.
5. **Errors at boundaries are milder.** If most errors sit near true contacts, the model places boundaries
   slightly off rather than calling whole beds wrong. Reference labels are also least certain at contacts.

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `Missing required input curves` | The file lacks one of GR, RDEP, RMED, DTC, RHOB. |
| `Could not load the trained models` | Check that the `models/` folder contains the `.joblib` and metadata files. |
| No original track or metrics | The label column is empty or unrecognised; pick it manually in ② Original lithology. |
| Page looks unstyled or old | Hard-refresh with `Cmd+Shift+R`; confirm `assets/style.css` exists. |
| Port already in use | `streamlit run app_v2.py --server.port 8502` |

## 7. Short description (for README / GitHub)

> **LITHO-ML Web Interface** — an interactive Streamlit application for lithology prediction from well logs.
> Upload a LAS or CSV well, run three frozen FORCE 2020 models (Ex-Tree, XGBoost, LightGBM), and inspect
> depth-aligned predictions, model agreement and confidence, confusion matrices, calibration, boundary-error
> analysis and petrophysical cross-plots. Metrics are computed only when reference labels exist, and the
> built-in Model guide documents each model's training and blind-test performance.
