# Active recognition models

`sign_classifier_canonical.pth` is the retained best static-letter checkpoint. `sequence_classifier.pth` is the separate J/Z dynamic checkpoint required for combined recognition. Keep their existing metrics and split manifests with them.

Matching browser ONNX exports remain in `output/python314-migration/onnx/models/`. The MediaPipe hand detector is stored separately in `asignx/assets/hand_landmarker.task`.

Unused experimental, manual and test checkpoints were removed at the owner's request. Their datasets, predictions and evaluation reports remain for reference. See [the cleanup inventory](../../output/model-cleanup/20261024/report.md).
