"""Compatibility entry point: fabricated recording sessions are no longer supported."""


def fix_sessions(csv_path: str = "ml/data/landmarks.csv", chunk_size: int = 50) -> None:
    raise ValueError(
        "Cannot turn row batches into independent recording sessions. "
        "The CSV was left unchanged. Re-import original images with "
        "ml.extract_dataset --manifest labels.csv, using real signer/origin groups, "
        "or collect additional independent recordings. See docs/extractor-guide.md."
    )


if __name__ == "__main__":
    fix_sessions()
