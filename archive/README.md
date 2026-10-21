# Historical data and model archive

`legacy-2026-10-20/legacy-files.zip` preserves three historical CSV snapshots and
the legacy 26-label model with its metrics and split manifest. A copy of the merge
audit is included; its original remains beside the active `ml/data/landmarks.csv`.
Each ZIP entry was verified against its original SHA-256 before removing the six
redundant working copies. `manifest.json` records file paths, sizes, checksums and
the unchanged active-file hashes.

The snapshot ending `backup-remove-J-Z-20261017T180240369474Z` exactly matches the
legacy checkpoint's dataset hash. It is retained for reproducing that historical
model, not for current canonical training.

To inspect or restore, extract to a separate folder first:

```powershell
Expand-Archive -LiteralPath archive/legacy-2026-10-20/legacy-files.zip -DestinationPath tmp/legacy-restore
```

The archive preserves project-relative paths. Copy back only explicitly needed
files after checking `manifest.json`; do not replace the current CSV with an old
snapshot. The archive stays inside this repository's workspace so no external
directory permissions are needed. It does not reduce the workspace's disk usage
by the entire original size because a compressed recovery copy is retained.

Static evaluation and training defaults now name
`ml/models/sign_classifier_canonical.pth`. Historical reports mentioning the old
`ml/models/sign_classifier.pth` refer to the archived model. The manual training
GUI's run-specific `sign_classifier.pth` filenames are separate versioned runs.
