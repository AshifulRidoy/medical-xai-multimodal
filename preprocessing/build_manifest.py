"""
preprocessing/build_manifest.py

Purpose:
    Unified manifest builder for CheXpert Plus and MIMIC-CXR.
    Reads `active_dataset` from config.yaml and routes to the correct loader.
    Outputs a manifest CSV with an identical schema regardless of source dataset:

        study_id | image_path | report_path | split | Atelectasis | ... (14 labels)

    Split indices are saved per-dataset to data/splits_{dataset}.json.

Inputs:
    - config.yaml  (active_dataset + per-dataset paths)

Outputs:
    - data/manifest_{dataset}.csv
    - data/splits_{dataset}.json

Example usage:
    python preprocessing/build_manifest.py --config config.yaml
    python preprocessing/build_manifest.py --config config.yaml --dataset chexpert_plus
    python preprocessing/build_manifest.py --config config.yaml --dataset mimic_cxr
"""

import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── Canonical label ordering (shared across all datasets) ──────────────
LABEL_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def apply_u_zeros(df: pd.DataFrame, label_cols: list) -> pd.DataFrame:
    """Replace uncertain labels (-1) and NaN with 0 (U-zeros policy).

    Args:
        df: DataFrame with label columns
        label_cols: Column names to process

    Returns:
        DataFrame with -1 and NaN replaced by 0
    """
    n_uncertain = (df[label_cols] == -1).sum().sum()
    df[label_cols] = df[label_cols].replace(-1, 0).fillna(0).astype(int)
    logger.info(f"U-zeros policy: replaced {n_uncertain} uncertain (-1) labels with 0")
    return df


def make_splits(
    df: pd.DataFrame, split_ratios: list, seed: int, splits_path: str
) -> pd.DataFrame:
    """Create stratified train/val/test splits and save to JSON.

    Args:
        df: Manifest DataFrame (must have Atelectasis and Pleural Effusion columns)
        split_ratios: [train, val, test] fractions summing to 1.0
        seed: Random seed
        splits_path: Output path for splits JSON

    Returns:
        DataFrame with a "split" column added
    """
    strat = df["Atelectasis"].astype(str) + "_" + df["Pleural Effusion"].astype(str)
    train_frac, val_frac, test_frac = split_ratios

    try:
        train_df, temp_df = train_test_split(
            df, train_size=train_frac, stratify=strat, random_state=seed
        )
        strat_temp = (
            temp_df["Atelectasis"].astype(str)
            + "_"
            + temp_df["Pleural Effusion"].astype(str)
        )
        val_size = val_frac / (val_frac + test_frac)
        val_df, test_df = train_test_split(
            temp_df, train_size=val_size, stratify=strat_temp, random_state=seed
        )
    except ValueError as e:
        logger.warning(f"Stratified split failed ({e}). Falling back to random split.")
        train_df, temp_df = train_test_split(
            df, train_size=train_frac, random_state=seed
        )
        val_size = val_frac / (val_frac + test_frac)
        val_df, test_df = train_test_split(
            temp_df, train_size=val_size, random_state=seed
        )

    train_df = train_df.copy()
    train_df["split"] = "train"
    val_df = val_df.copy()
    val_df["split"] = "val"
    test_df = test_df.copy()
    test_df["split"] = "test"
    out = pd.concat([train_df, val_df, test_df], ignore_index=True)

    logger.info(
        f"Splits → train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    splits = {
        "train": train_df["study_id"].tolist(),
        "val": val_df["study_id"].tolist(),
        "test": test_df["study_id"].tolist(),
    }
    with open(splits_path, "w") as f:
        json.dump(splits, f)
    logger.info(f"Split indices saved to {splits_path}")
    return out


# ══════════════════════════════════════════════
# LOADER: CheXpert Plus
# ══════════════════════════════════════════════
def build_chexpert_plus(config: dict) -> pd.DataFrame:
    """Build manifest from CheXpert Plus (Stanford AIMI).

    CSV layout (df_chexpert_plus_240401.csv, 223,462 rows x 27 columns):
        path_to_image       - relative path: train/patientXXX/studyY/viewZ.jpg
        frontal_lateral     - "Frontal" | "Lateral"
        ap_pa               - "AP" | "PA"
        deid_patient_id     - e.g. "patient42142"
        patient_report_date_order - study index for patient (used as study_id)
        report              - full raw report text (inline)
        section_findings    - pre-extracted Findings section
        section_impression  - pre-extracted Impression section
        split               - "train" | "valid" (pre-defined by Stanford)
        age, sex, race, ...  - demographic columns (available for analysis)

    Labels come from chexbert_labels.zip (separate file), which must be
    downloaded and extracted alongside the main CSV. The zip contains a CSV
    with columns: study_id, Atelectasis, Cardiomegaly, ... (14 classes).

    Args:
        config: Full config dict from config.yaml

    Returns:
        Manifest DataFrame with unified schema
    """
    paths = config["paths"]["chexpert_plus"]
    main_csv = paths["main_csv"]
    labels_csv = paths["chexbert_labels_csv"]
    images_dir = paths["images_dir"]
    n_studies = config["dataset"]["n_studies"]
    seed = config["dataset"]["seed"]

    # ── Validate required files ────────────────────────────────────────
    for fpath, hint in [
        (
            main_csv,
            "Download df_chexpert_plus_240401.csv via azcopy from the AIMI container",
        ),
        (labels_csv, "Extract chexbert_labels.zip from the AIMI container"),
    ]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"CheXpert Plus file not found: {fpath}\n{hint}\n"
                "See data/README.md for full instructions."
            )

    # ── Load main CSV ──────────────────────────────────────────────────
    logger.info(f"Loading CheXpert Plus main CSV: {main_csv}")
    df = pd.read_csv(main_csv)
    logger.info(f"CheXpert Plus: {len(df):,} rows, {len(df.columns)} columns")

    # ── Filter to frontal views ────────────────────────────────────────
    before = len(df)
    df = df[df["frontal_lateral"] == "Frontal"].copy()
    logger.info(
        f"Frontal filter: {len(df):,} rows (dropped {before - len(df):,} lateral views)"
    )

    # ── Build stable study_id from patient + study order ──────────────
    # deid_patient_id = "patient42142", patient_report_date_order = 5
    # → study_id = "patient42142_study5"
    df["study_id"] = (
        df["deid_patient_id"].astype(str)
        + "_study"
        + df["patient_report_date_order"].astype(str)
    )

    # ── Build absolute image paths ─────────────────────────────────────
    # path_to_image is relative: "train/patient42142/study5/view1_frontal.jpg"
    df["image_path"] = df["path_to_image"].apply(
        lambda p: os.path.join(images_dir, str(p))
    )

    # ── Build report text: prefer section_findings + section_impression ─
    # Write each study's report to a .txt file for uniform text_cleaning input.
    reports_dir = os.path.join(os.path.dirname(main_csv), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    def build_report_txt(row) -> str:
        """Combine findings + impression into a .txt file; return path."""
        rpt_path = os.path.join(reports_dir, f"{row['study_id']}.txt")
        if not os.path.exists(rpt_path):
            findings = (
                str(row["section_findings"])
                if pd.notna(row["section_findings"])
                else ""
            )
            impression = (
                str(row["section_impression"])
                if pd.notna(row["section_impression"])
                else ""
            )
            # Fall back to full report if both sections are empty
            if not findings and not impression:
                raw = str(row["report"]) if pd.notna(row["report"]) else ""
            else:
                raw = ""
            with open(rpt_path, "w") as f:
                if findings:
                    f.write(f"FINDINGS:\n{findings.strip()}\n\n")
                if impression:
                    f.write(f"IMPRESSION:\n{impression.strip()}\n")
                if raw:
                    f.write(raw.strip())
        return rpt_path

    logger.info("Writing per-study report .txt files (first run only)...")
    df["report_path"] = df.apply(build_report_txt, axis=1)

    # ── Load and merge CheXbert labels ────────────────────────────────
    # Labels come as three JSON files keyed on path_to_image:
    #   report_fixed.json    — labelled from full report   (primary)
    #   impression_fixed.json — labelled from impression   (secondary fill)
    #   findings_fixed.json   — labelled from findings     (tertiary fill)
    # null values mean uncertain; U-zeros policy converts them to 0.
    # Strategy: start with report labels, fill remaining nulls from impression,
    # then from findings. This maximises label coverage per study.
    labels_dir = os.path.dirname(labels_csv)  # reuse as the json directory

    def load_label_json(fname: str) -> pd.DataFrame:
        """Load a CheXbert label JSON file into a DataFrame keyed on path_to_image."""
        fpath = os.path.join(labels_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"CheXbert label file not found: {fpath}\n"
                "Extract chexbert_labels.zip from the AIMI container. "
                "See data/README.md."
            )
        ldf = pd.read_json(fpath, lines=True)
        logger.info(f"  {fname}: {len(ldf):,} rows")
        return ldf

    logger.info("Loading CheXbert label files...")
    rpt_df = load_label_json("report_fixed.json")
    imp_df = load_label_json("impression_fixed.json")
    fnd_df = load_label_json("findings_fixed.json")

    # Merge all three on path_to_image with suffixes to track source
    combined = rpt_df.set_index("path_to_image")
    for src_df, suffix in [(imp_df, "_imp"), (fnd_df, "_fnd")]:
        src = src_df.set_index("path_to_image")
        for cls in LABEL_CLASSES:
            if cls in combined.columns and cls in src.columns:
                # Fill nulls in primary with values from secondary source
                combined[cls] = combined[cls].combine_first(src[cls])

    combined = combined.reset_index().rename(
        columns={"path_to_image": "path_to_image_label"}
    )

    # Merge into main df on path_to_image
    df = df.merge(
        combined[
            ["path_to_image_label"]
            + [c for c in LABEL_CLASSES if c in combined.columns]
        ],
        left_on="path_to_image",
        right_on="path_to_image_label",
        how="left",
    ).drop(columns=["path_to_image_label"], errors="ignore")

    for cls in LABEL_CLASSES:
        if cls not in df.columns:
            df[cls] = 0

    # Apply U-zeros policy (null → 0, any -1 → 0)
    df = apply_u_zeros(df, LABEL_CLASSES)

    # ── Subsample if needed ────────────────────────────────────────────
    if len(df) > n_studies:
        strat = df["Atelectasis"].astype(str) + "_" + df["Pleural Effusion"].astype(str)
        try:
            df, _ = train_test_split(
                df, train_size=n_studies, stratify=strat, random_state=seed
            )
        except ValueError:
            df = df.sample(n=n_studies, random_state=seed)
        logger.info(f"CheXpert Plus: subsampled to {len(df):,} studies")

    keep_cols = ["study_id", "image_path", "report_path"] + LABEL_CLASSES
    return df[keep_cols].reset_index(drop=True)


# ══════════════════════════════════════════════
# LOADER: MIMIC-CXR
# ══════════════════════════════════════════════
def build_mimic_cxr(config: dict) -> pd.DataFrame:
    """Build manifest from MIMIC-CXR-JPG (PhysioNet).

    Requires credentialed PhysioNet access. Filters to PA views, subsamples
    to n_studies with stratification, and constructs image and report paths.

    Args:
        config: Full config dict from config.yaml

    Returns:
        Manifest DataFrame with unified schema
    """
    paths = config["paths"]["mimic_cxr"]
    n_studies = config["dataset"]["n_studies"]
    seed = config["dataset"]["seed"]

    for key in ["chexpert_labels", "split_csv", "metadata_csv"]:
        if not os.path.exists(paths[key]):
            raise FileNotFoundError(
                f"MIMIC-CXR file not found: {paths[key]}\n"
                "Requires PhysioNet credentialing. See data/README.md."
            )

    labels_df = pd.read_csv(paths["chexpert_labels"])
    split_df = pd.read_csv(paths["split_csv"])
    meta_df = pd.read_csv(paths["metadata_csv"])

    logger.info(f"MIMIC-CXR: {len(labels_df):,} labels, {len(meta_df):,} metadata rows")

    # Filter to PA (frontal) views
    pa_meta = meta_df[meta_df["ViewPosition"] == "PA"].copy()
    logger.info(f"MIMIC-CXR: {len(pa_meta):,} PA views")

    merged = pa_meta.merge(
        split_df[["dicom_id", "subject_id", "study_id"]], on="dicom_id", how="inner"
    )
    # One image per study
    merged = merged.sort_values("dicom_id").groupby("study_id").first().reset_index()
    merged = merged.merge(
        labels_df[["subject_id", "study_id"] + LABEL_CLASSES],
        on=["subject_id", "study_id"],
        how="inner",
    )
    logger.info(f"MIMIC-CXR: {len(merged):,} unique PA studies with labels")

    merged = apply_u_zeros(merged, LABEL_CLASSES)

    if len(merged) > n_studies:
        strat = (
            merged["Atelectasis"].astype(str)
            + "_"
            + merged["Pleural Effusion"].astype(str)
        )
        try:
            merged, _ = train_test_split(
                merged, train_size=n_studies, stratify=strat, random_state=seed
            )
        except ValueError:
            merged = merged.sample(n=n_studies, random_state=seed)
        logger.info(f"MIMIC-CXR: subsampled to {len(merged):,} studies")

    images_dir = paths["images_dir"]
    raw_dir = paths["raw_dir"]

    def img_path(row):
        sid = str(row["subject_id"])
        return os.path.join(
            images_dir,
            f"p{sid[:2]}",
            f"p{sid}",
            f"s{row['study_id']}",
            f"{row['dicom_id']}.jpg",
        )

    def rpt_path(row):
        sid = str(row["subject_id"])
        return os.path.join(
            raw_dir,
            "files",
            f"p{sid[:2]}",
            f"p{sid}",
            f"s{row['study_id']}",
            f"s{row['study_id']}.txt",
        )

    merged["image_path"] = merged.apply(img_path, axis=1)
    merged["report_path"] = merged.apply(rpt_path, axis=1)
    merged["study_id"] = merged["study_id"].astype(str)

    keep_cols = ["study_id", "image_path", "report_path"] + LABEL_CLASSES
    return merged[keep_cols].reset_index(drop=True)


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
LOADERS = {
    "chexpert_plus": build_chexpert_plus,
    "mimic_cxr": build_mimic_cxr,
}


def main() -> None:
    """Entry point — parse args and build the manifest for the active dataset."""
    parser = argparse.ArgumentParser(
        description="Build manifest CSV for CheXpert Plus or MIMIC-CXR."
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override active_dataset from config (chexpert_plus | mimic_cxr)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset = args.dataset or config["active_dataset"]
    if dataset not in LOADERS:
        logger.error(f"Unknown dataset: '{dataset}'. Choose from: {list(LOADERS)}")
        sys.exit(1)

    seed = config["dataset"]["seed"]
    set_seed(seed)

    logger.info(f"Building manifest for dataset: {dataset}")

    df = LOADERS[dataset](config)

    dataset_paths = config["paths"][dataset]
    splits_path = dataset_paths["splits"]
    manifest_path = dataset_paths["manifest"]

    os.makedirs(os.path.dirname(splits_path) or ".", exist_ok=True)
    df = make_splits(df, config["dataset"]["split_ratios"], seed, splits_path)

    if "report_path" not in df.columns:
        df["report_path"] = ""

    keep_cols = ["study_id", "image_path", "report_path", "split"] + LABEL_CLASSES
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    df.to_csv(manifest_path, index=False)
    logger.info(f"Manifest saved to {manifest_path} ({len(df):,} rows)")

    # Label distribution summary
    logger.info(f"\n=== Label Distribution [{dataset}] ===")
    for label in LABEL_CLASSES:
        if label in df.columns:
            pos = int(df[label].sum())
            pct = pos / len(df) * 100
            logger.info(f"  {label:<35} {pos:>5} positive ({pct:.1f}%)")

    has_reports = (df["report_path"] != "").sum()
    logger.info(f"\nStudies with report text: {has_reports:,} / {len(df):,}")

    logger.info(f"\n✓ Done. Next step:")
    if dataset == "chexpert_plus":
        logger.info(
            "  python models/train_cnn.py --config config.yaml --run_name cnn_chexpert_plus_v1"
        )
    else:
        logger.info(
            "  python models/train_cnn.py --config config.yaml --run_name cnn_mimic_v1"
        )


if __name__ == "__main__":
    main()
