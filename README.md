# Explainable Multi-Modal AI for Clinical Diagnosis from Chest X-rays

**Research-grade implementation** 

## Abstract

We present a multi-modal explainable AI pipeline for chest pathology detection from MIMIC-CXR chest X-rays and radiology reports. The system trains a CNN baseline (ResNet-18) with Grad-CAM explainability, fine-tunes a large vision-language model (LiquidAI/LFM2.5-VL-1.6B) via LoRA as a teacher, and distills the teacher into a lightweight deployable student using knowledge distillation. Evaluation covers ROC-AUC, F1, calibration (ECE), and qualitative Grad-CAM analysis across 14 CheXpert pathology classes. This pipeline is designed for reproducible research on constrained hardware (16GB Apple M4) and is suitable as a portfolio project or workshop submission on explainable medical AI.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MIMIC-CXR Dataset                           │
│              (Chest X-rays + Radiology Reports + Labels)            │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
           ┌────────────▼──────────────┐
           │     Data Pipeline         │
           │  build_manifest.py        │
           │  image_pipeline.py        │
           │  text_cleaning.py         │
           └────────────┬──────────────┘
                        │
        ┌───────────────┴────────────────┐
        │                                │
┌───────▼────────┐             ┌─────────▼──────────┐
│  Stage 1       │             │  Stage 2            │
│  CNN Baseline  │             │  LoRA Teacher       │
│  ResNet-18     │             │  LFM2.5-VL-1.6B     │
│  + Grad-CAM    │             │  (Text → Diagnosis) │
└───────┬────────┘             └─────────┬───────────┘
        │                                │
        │                     ┌──────────▼──────────┐
        │                     │  Stage 3            │
        │                     │  Knowledge          │
        │                     │  Distillation       │
        │                     │  Teacher → Student  │
        │                     └──────────┬──────────┘
        │                                │
        └──────────────┬─────────────────┘
                       │
              ┌────────▼────────┐
              │   Evaluation    │
              │   + Analysis    │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │   Streamlit     │
              │   Demo App      │
              └─────────────────┘
```

---

## Installation

### 1. Clone and set up environment

```bash
git clone https://github.com/your-username/medical-xai-multimodal.git
cd medical-xai-multimodal
python -m venv venv && source venv/bin/activate

# Install PyTorch for Apple Silicon (MPS backend)
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# Install all remaining dependencies
pip install -r requirements.txt
```

### 2. Download MIMIC-CXR data

See `data/README.md` for full instructions. You will need a credentialed PhysioNet account.

```bash
# After obtaining PhysioNet access, download required files
# See data/README.md for exact wget commands
```

---

## Running the Pipeline

All stages are independently runnable via CLI. Run in order.

### Stage 0: Setup

```bash
# Verify directory structure and config
cat config.yaml
```

### Stage 1: Build Data Manifest

```bash
python preprocessing/build_manifest.py \
    --config config.yaml \
    --n_studies 15000
```

**Output:** `data/manifest.csv`, `data/splits.json`

### Stage 2: Train CNN Baseline

```bash
python models/train_cnn.py \
    --config config.yaml \
    --run_name cnn_baseline_v1
```

**Output:** `outputs/cnn_baseline_v1/best_model.pt`

> ⚡ **Checkpoint:** Verify val_auc ≥ 0.80 before proceeding.

### Stage 3: Evaluate CNN + Generate Grad-CAM

```bash
python evaluation/evaluate_cnn.py \
    --config config.yaml \
    --run_name cnn_baseline_v1
```

**Output:** `outputs/cnn_baseline_v1/test_results.json`

### Stage 4: LoRA Teacher Fine-Tuning (Phase 1: Text Only)

```bash
python models/lfm_finetune.py \
    --config config.yaml \
    --run_name lora_teacher_phase1
```

**Output:** `outputs/lora_teacher_phase1/adapter_weights/`

> ⚠️ Requires ~10–13 GB unified RAM. Close all other applications.

> ⚡ **Checkpoint:** Verify teacher outperforms CNN before proceeding.

### Stage 5: Cache Teacher Soft Labels

```bash
python models/cache_teacher_outputs.py \
    --config config.yaml \
    --teacher_run lora_teacher_phase1
```

**Output:** `data/teacher_soft_labels.npy`, `data/teacher_study_ids.json`

### Stage 6: Knowledge Distillation

```bash
python models/distillation.py \
    --config config.yaml \
    --run_name student_v1
```

**Output:** `outputs/student_v1/best_student.pt`

> ⚡ **Checkpoint:** Verify student retains ≥ 90% of teacher AUC.

### Stage 7: Final Comparison

```bash
python evaluation/compare_models.py \
    --config config.yaml \
    --cnn_run cnn_baseline_v1 \
    --teacher_run lora_teacher_phase1 \
    --student_run student_v1
```

**Output:** `outputs/final_comparison.json`, `outputs/final_comparison.csv`, `outputs/calibration_comparison.png`

### Stage 8: Failure Analysis

```bash
python evaluation/failure_analysis.py \
    --config config.yaml \
    --student_run student_v1
```

**Output:** `outputs/failure_analysis/summary.md`

### Stage 9: Streamlit Demo

```bash
streamlit run demo_app/streamlit_app.py
```

---

## Results

*Metrics will be populated after experiments complete. Values below are placeholders.*

| Metric | CNN Baseline | Teacher (LFM LoRA) | Distilled Student |
|---|---|---|---|
| Parameters | ~11M | ~1.6B (6M trainable) | ~11M |
| ROC-AUC (macro) | TBD | TBD | TBD |
| F1-score (macro) | TBD | TBD | TBD |
| Sensitivity | TBD | TBD | TBD |
| Specificity | TBD | TBD | TBD |
| ECE | TBD | TBD | TBD |
| Inference time (ms/sample) | TBD | TBD | TBD |
| Knowledge retained vs. Teacher | — | 100% | target ≥ 90% |

> **Reproducibility note:** All numbers in this table must be produced by running `evaluation/compare_models.py` with the checkpoints in `outputs/`. No manually entered values.

---

## Limitations and Ethical Considerations

### Dataset Demographic Bias

MIMIC-CXR originates from Beth Israel Deaconess Medical Center (Boston, MA, USA). The patient population is predominantly from a single US hospital system and may not generalize well to populations with different demographics, comorbidity profiles, or imaging equipment. Models trained on this data may underperform on radiographs from different hospitals or regions without domain adaptation.

### Label Noise from Automated Labeling

Pathology labels are derived from the CheXpert labeler — an NLP system applied to free-text radiology reports. This introduces systematic label noise:
- The U-zeros policy (treating uncertain `-1` labels as negative) may suppress recall for rare pathologies.
- CheXpert labeling accuracy varies by pathology class (e.g., Fracture and Lung Lesion are known to be harder to extract from text).
- All reported metrics should be interpreted with this labeling uncertainty in mind.

### Domain Shift

Performance on radiographs acquired outside of MIMIC-CXR's equipment, protocols, and patient population is unknown. Significant degradation should be expected without fine-tuning or domain adaptation.

### Overconfidence Risks in Clinical Settings

Neural networks, particularly after distillation, may be poorly calibrated. ECE is reported for all models. Any deployment of such a system in a clinical workflow would require:
- Prospective validation on an independent cohort
- Uncertainty quantification and abstention mechanisms
- Regulatory review (FDA 510(k) or equivalent)

### NOT for Clinical Use

**This system is a research prototype. It has not been validated for clinical use, does not carry any regulatory approval, and must never be used to inform real medical decisions.** All model outputs require review by a qualified, licensed radiologist.

---

## Citation

If you use this codebase, please cite the MIMIC-CXR dataset:

```bibtex
@article{johnson2019mimic,
  title={MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports},
  author={Johnson, Alistair EW and Pollard, Tom J and Berkowitz, Seth J and others},
  journal={Scientific data},
  volume={6},
  number={1},
  pages={317},
  year={2019},
  publisher={Nature Publishing Group}
}
```

---

## Project Structure

```
medical-xai-multimodal/
├── config.yaml
├── requirements.txt
├── README.md
├── data/
│   ├── README.md              ← MIMIC-CXR download guide
│   ├── splits.json
│   └── manifest.csv
├── preprocessing/
│   ├── build_manifest.py
│   ├── image_pipeline.py
│   └── text_cleaning.py
├── models/
│   ├── cnn_baseline.py
│   ├── train_cnn.py
│   ├── lfm_finetune.py
│   ├── cache_teacher_outputs.py
│   └── distillation.py
├── explainability/
│   ├── grad_cam.py
│   └── findings.md            ← Clinical commentary template
├── evaluation/
│   ├── metrics.py
│   ├── evaluate_cnn.py
│   ├── compare_models.py
│   └── failure_analysis.py
├── outputs/                   ← All experiment results (gitignored)
└── demo_app/
    └── streamlit_app.py
```

---

