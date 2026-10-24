# Model cleanup ? 24 October 2026

Deleted 232 unused model/export files (23,346,151 bytes). This includes obsolete experiment/manual checkpoints, old exports and disposable test artifacts. Only regular files explicitly listed in the hashed plan were removed; no recursive directory deletion was used.

## Retained

- Current best static checkpoint: `ml/models/sign_classifier_canonical.pth`.
- Required J/Z checkpoint: `ml/models/sequence_classifier.pth`.
- Their matching ONNX graphs and external weights in `output/python314-migration/onnx/models/`, with existing metadata.
- Both hand detector task assets, including the packaged extractor copy.

Current static and sequence checkpoints loaded successfully. Browser static inference matched PyTorch within 2.39e-6 maximum logit error; J/Z single-stream inference matched within 5.97e-7. The existing sequence ONNX export failed a batch-of-four check; its supported single-stream input passed. No exporter behavior was changed by this cleanup.

All retained model SHA-256 hashes and the source dataset were checked unchanged after deletion. Datasets, split manifests, training metrics, prediction CSVs and reports were retained. Experiment pointers are marked `checkpoint_retained: false` where applicable.

[Exact retained/deleted file inventory and hashes](plan.json), [cleanup verification](result.json).

## Updated experiment pointers

- `output\added-M-reevaluation\latest-run.json`
- `output\full-static-retraining\latest-run.json`
- `output\source-separated-training\latest-run.json`
- `output\whole-model-finetuning\latest-run.json`
- `output\source-separated-training\20261024-source-holdout\status.json`
