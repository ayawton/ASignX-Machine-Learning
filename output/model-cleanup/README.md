# Retained models

Unused experimental checkpoint files were removed at the user's request. Historical datasets, split manifests, metrics, and reports remain; references to removed experimental checkpoints describe past evaluations and are not runnable model artifacts.

Retained application checkpoints:

- `ml/models/sign_classifier_canonical.pth`: selected older static-sign model. Test macro F1 63.07% on the 34,720-sample cleaned evaluation set; 59.49% on the original test set.
- `ml/models/sequence_classifier.pth`: J/Z sequence model used by `ASignX_TestRecognition.bat`.

Both retained checkpoints have their original hashes. The JSON cleanup record lists the paths and hashes of the ten deleted checkpoints. Training-run status files identify their model artifacts as removed. No dataset samples or evaluation scores were changed by this cleanup.
