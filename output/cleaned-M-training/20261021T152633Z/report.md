# Cleaned dataset training and evaluation

| Metric | Retained model | Retrained model |
|---|---:|---:|
| Validation macro F1 | 97.33% | 97.72% |
| Test macro F1 | 63.07% | 57.77% |
| Test accuracy | 56.62% | 49.57% |
| Test weighted F1 | 59.42% | 51.53% |
| M test F1 | 73.06% | 68.14% |
| N test F1 | 56.49% | 56.20% |

Trained for 50 epochs; selected epoch 43 using validation only. The test contains 34,720 samples, validation 13,791, and training 109,223. The existing deployed checkpoint is preserved.

Removed rows belonged only to test. Training and validation data are unchanged. Compare both models on this same cleaned test; differences from historical test scores include a test-composition change.

Offline static landmark classification on a development-known session split; signer independence unverified. Scores are unthresholded and do not measure camera detection or live gesture recognition.
