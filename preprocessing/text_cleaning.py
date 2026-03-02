"""
preprocessing/text_cleaning.py

Purpose:
    Load, clean, and tokenize MIMIC-CXR radiology reports.
    Extracts FINDINGS and IMPRESSION sections, removes PHI placeholders,
    boilerplate, and extra whitespace. Provides a ReportDataset class
    for lazy loading paired reports and labels.

Inputs:
    - Radiology report .txt files (paths from manifest.csv)
    - HuggingFace tokenizer instance

Outputs:
    - Cleaned text strings
    - Tokenized input_ids and attention_masks
    - ReportDataset for use in LoRA fine-tuning

Example usage:
    from preprocessing.text_cleaning import load_report, clean_text, extract_sections
    text = load_report("data/raw/files/p10/p10000032/s50414267/s50414267.txt")
    findings, impression = extract_sections(text)
    cleaned = clean_text(findings + " " + impression)
"""

import logging
import os
import re
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Label classes (canonical ordering)
# ──────────────────────────────────────────────
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

# Regex patterns for cleaning
_PHI_PATTERN = re.compile(r"\[\*\*.*?\*\*\]")
_EXTRA_WHITESPACE = re.compile(r"[ \t]+")
_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
_SECTION_HEADER = re.compile(
    r"^(RADIOLOGY REPORT|FINAL REPORT|EXAMINATION|INDICATION|COMPARISON|TECHNIQUE"
    r"|WET READ|RECOMMENDATION|ADDENDUM|CLINICAL INFORMATION|HISTORY).*?$",
    re.IGNORECASE | re.MULTILINE,
)


def load_report(report_path: str) -> str:
    """Read a MIMIC-CXR radiology report text file.

    Args:
        report_path: Absolute or relative path to the .txt report file

    Returns:
        Raw text content of the report, or empty string if file not found
    """
    try:
        with open(report_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"Report not found: {report_path}")
        return ""
    except OSError as e:
        logger.warning(f"Cannot read report {report_path}: {e}")
        return ""


def extract_sections(text: str) -> Tuple[str, str]:
    """Extract FINDINGS and IMPRESSION sections from radiology report text.

    Searches case-insensitively for section headers. Returns empty strings
    if sections are not found.

    Args:
        text: Raw radiology report text

    Returns:
        Tuple of (findings_text, impression_text)
        Both may be empty strings if sections are absent.
    """
    findings = ""
    impression = ""

    # Pattern: section header followed by content until the next section or end
    findings_match = re.search(
        r"FINDINGS?[:.\s]+(.*?)(?=\n[A-Z][A-Z ]+:|IMPRESSION|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    impression_match = re.search(
        r"IMPRESSION[:.\s]+(.*?)(?=\n[A-Z][A-Z ]+:|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    if findings_match:
        findings = findings_match.group(1).strip()
    if impression_match:
        impression = impression_match.group(1).strip()

    # Fallback: if neither section found, use full text (some reports lack headers)
    if not findings and not impression:
        findings = text.strip()

    return findings, impression


def clean_text(text: str) -> str:
    """Clean radiology report text for model input.

    Removes:
    - PHI placeholders like [** ... **]
    - Boilerplate section headers
    - Excessive whitespace and blank lines
    - Leading/trailing whitespace

    Args:
        text: Raw or partially extracted text

    Returns:
        Cleaned text string
    """
    if not text:
        return ""

    # Remove PHI placeholders
    text = _PHI_PATTERN.sub("", text)

    # Remove boilerplate section headers
    text = _SECTION_HEADER.sub("", text)

    # Normalize whitespace
    text = _EXTRA_WHITESPACE.sub(" ", text)
    text = _MULTIPLE_NEWLINES.sub("\n\n", text)

    # Remove lines that are only whitespace or punctuation
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if len(line) > 2]
    text = " ".join(lines)

    return text.strip()


def prepare_report_text(report_path: str) -> str:
    """Full pipeline: load → extract sections → clean.

    Args:
        report_path: Path to .txt report file

    Returns:
        Cleaned combined findings + impression text
    """
    raw = load_report(report_path)
    findings, impression = extract_sections(raw)
    combined = f"{findings} {impression}".strip()
    return clean_text(combined)


def tokenize_report(
    text: str,
    tokenizer,
    max_length: int = 512,
) -> dict:
    """Tokenize cleaned report text using a HuggingFace tokenizer.

    Args:
        text: Cleaned report text string
        tokenizer: HuggingFace PreTrainedTokenizer instance
        max_length: Maximum token sequence length

    Returns:
        Dict with keys "input_ids" and "attention_mask" as torch.Tensor (1D)
    """
    if not text:
        # Return a single [PAD] token for empty reports
        text = tokenizer.pad_token or "[PAD]"

    encoded = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    # Squeeze batch dimension: [1, max_length] → [max_length]
    return {
        "input_ids": encoded["input_ids"].squeeze(0),
        "attention_mask": encoded["attention_mask"].squeeze(0),
    }


class ReportDataset(Dataset):
    """Lazy-loading PyTorch Dataset for radiology report text.

    Returns (input_ids, attention_mask, label_tensor, study_id) for each sample.
    Handles missing report files gracefully (returns empty token sequence).

    Args:
        manifest_path: Path to data/manifest.csv
        split: One of "train", "val", "test"
        tokenizer: HuggingFace tokenizer
        max_length: Maximum token length
        label_classes: Ordered list of pathology label column names
    """

    def __init__(
        self,
        manifest_path: str,
        split: str,
        tokenizer,
        max_length: int = 512,
        label_classes: Optional[list] = None,
    ) -> None:
        """Initialize dataset."""
        assert os.path.exists(manifest_path), (
            f"Manifest not found: {manifest_path}. "
            "Run preprocessing/build_manifest.py first."
        )
        assert split in ("train", "val", "test"), f"Invalid split: {split}"

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_classes = label_classes or LABEL_CLASSES

        df = pd.read_csv(manifest_path)
        self.data = df[df["split"] == split].reset_index(drop=True)
        assert len(self.data) > 0, f"No samples for split='{split}'"
        logger.info(f"ReportDataset [{split}]: {len(self.data)} samples")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """Load and tokenize one report.

        Args:
            idx: Integer index

        Returns:
            Tuple of (input_ids, attention_mask, label_tensor, study_id)
        """
        row = self.data.iloc[idx]
        study_id = str(row["study_id"])
        report_path = row["report_path"]

        # Load and clean report
        text = prepare_report_text(report_path)

        # Tokenize
        tokens = tokenize_report(text, self.tokenizer, self.max_length)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # Labels
        label_values = row[self.label_classes].values.astype("float32")
        label_tensor = torch.tensor(label_values, dtype=torch.float32)

        return input_ids, attention_mask, label_tensor, study_id


# ──────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with a synthetic report string
    dummy_report = """
    EXAMINATION: Chest PA and lateral

    INDICATION: [**Age 65**] year old with shortness of breath.

    COMPARISON: Prior study from [**2020-01-15**].

    FINDINGS: The lungs are clear bilaterally. No focal consolidation,
    pleural effusion, or pneumothorax is identified. The cardiac silhouette
    is mildly enlarged. There is no vascular congestion.

    IMPRESSION:
    1. Mild cardiomegaly.
    2. No acute cardiopulmonary process.
    """

    findings, impression = extract_sections(dummy_report)
    print(f"Findings: {findings[:80]}...")
    print(f"Impression: {impression[:80]}...")

    combined = f"{findings} {impression}"
    cleaned = clean_text(combined)
    print(f"Cleaned ({len(cleaned)} chars): {cleaned[:120]}...")

    # Test PHI removal
    phi_text = "Patient [**John Smith**] age [**45**] seen on [**2021-03-15**]."
    cleaned_phi = clean_text(phi_text)
    assert "[**" not in cleaned_phi, "PHI not removed!"
    print(f"PHI removed: '{cleaned_phi}'")

    print("✓ text_cleaning smoke-test passed.")
