# Data Setup Guide — MIMIC-CXR

This project uses the **MIMIC-CXR-JPG** dataset, which provides chest X-ray images, radiology reports, and structured pathology labels derived from the CheXpert labeler.

**Access requires credentialed PhysioNet approval. Do not attempt to download without completing this process.**

---

## Step 1: Obtain PhysioNet Credentials

1. Create a free account at [https://physionet.org/register/](https://physionet.org/register/)
2. Complete the CITI "Data or Specimens Only Research" training course
3. Go to [MIMIC-CXR-JPG on PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) and click **"Request access"**
4. Wait for approval email (typically 1–3 business days)

---

## Step 2: Download the Dataset

After approval, use `wget` with your PhysioNet credentials:

```bash
# Create destination directory
mkdir -p data/raw && cd data/raw

# Download the full dataset (~250GB) — or use the subset approach below
wget -r -N -c -np \
  --user YOUR_PHYSIONET_USERNAME \
  --ask-password \
  https://physionet.org/files/mimic-cxr-jpg/2.0.0/
```

**Recommended: Partial download for this project's 15k study subset**

You only need:
- `mimic-cxr-2.0.0-chexpert.csv` — CheXpert-derived labels
- `mimic-cxr-2.0.0-split.csv` — Official train/validate/test split
- `mimic-cxr-2.0.0-metadata.csv` — Image metadata (view position, etc.)
- `files/p*/p*/s*/*.jpg` — The actual JPEG images
- `files/p*/p*/s*/*.txt` — The radiology report text files

```bash
# Download metadata files only first
wget -N --user YOUR_USERNAME --ask-password \
  https://physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv \
  https://physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv \
  https://physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv
```

---

## Step 3: Expected Directory Layout

After download, your `data/raw/` directory should look like:

```
data/raw/
├── mimic-cxr-2.0.0-chexpert.csv      # Labels (14 pathology classes)
├── mimic-cxr-2.0.0-split.csv         # Official train/validate/test assignment
├── mimic-cxr-2.0.0-metadata.csv      # Image-level metadata (ViewPosition, etc.)
└── files/
    └── p10/
        └── p10000032/
            └── s50414267/
                ├── 02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
                └── s50414267.txt      # Radiology report
```

The image paths follow: `files/p{patient_id[:3]}/{patient_id}/{study_id}/{dicom_id}.jpg`

---

## Step 4: Verify Data Is in Place

Run the following to confirm all required files exist before proceeding:

```bash
python - <<'EOF'
import os, sys

required_files = [
    "data/raw/mimic-cxr-2.0.0-chexpert.csv",
    "data/raw/mimic-cxr-2.0.0-split.csv",
    "data/raw/mimic-cxr-2.0.0-metadata.csv",
    "data/raw/files",
]

all_ok = True
for f in required_files:
    exists = os.path.exists(f)
    status = "✓" if exists else "✗ MISSING"
    print(f"  {status}  {f}")
    if not exists:
        all_ok = False

if all_ok:
    import pandas as pd
    df = pd.read_csv("data/raw/mimic-cxr-2.0.0-chexpert.csv")
    print(f"\n  Studies with labels: {len(df):,}")
    print("\n  ✓ Data verification passed. Run build_manifest.py next.")
else:
    print("\n  ✗ Some files are missing. Please complete the download.")
    sys.exit(1)
EOF
```

---

## Key CSV Files

### `mimic-cxr-2.0.0-chexpert.csv`

| Column | Description |
|---|---|
| `subject_id` | Patient identifier |
| `study_id` | Study (visit) identifier |
| `Atelectasis` | 1 = positive, 0 = negative, -1 = uncertain, NaN = not mentioned |
| ... | (14 pathology columns total) |

### `mimic-cxr-2.0.0-split.csv`

| Column | Description |
|---|---|
| `dicom_id` | Individual image identifier |
| `subject_id` | Patient identifier |
| `study_id` | Study identifier |
| `split` | `train`, `validate`, or `test` |

### `mimic-cxr-2.0.0-metadata.csv`

| Column | Description |
|---|---|
| `dicom_id` | Image identifier |
| `ViewPosition` | `PA`, `AP`, `LL`, etc. — we use `PA` only |

---

## Privacy and Compliance

- MIMIC-CXR is **de-identified**. No real PHI is present.
- However, you must sign the PhysioNet data use agreement before downloading.
- **Never commit any data files to version control.** The `data/raw/` directory is in `.gitignore`.
- Never log patient identifiers (`subject_id`) in model outputs or experiment logs.
