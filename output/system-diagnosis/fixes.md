# Live-state fixes and benchmark preparation — 24 October 2026

> Subsequent user-requested diagnostic: [whole-model fine-tuning completed](../whole-model-finetuning/20261023T204455Z/report.md) using the historical split. Signer-separated benchmark readiness remains false.

The source fixes address the reproduced behaviors in the [diagnosis](report.md). Historical model scores in that diagnosis are unchanged; this work changes live handling and evaluation tooling, not trained weights.

## Implemented

- Camera captures have sequence IDs and monotonic capture timestamps. Threaded reads consume each capture once, timeout notifications cannot reuse cached frames, failed reads invalidate availability, and two seconds without a successful capture raises a clear camera error. The static studio waits normally between fresh frames.
- Main recognition and standalone dynamic prediction use capture timestamps. Static and unified dynamic confirmation restart after a long input gap.
- Left-hand motion recovery only repairs uncertain handedness predictions; high-confidence opposite-hand detections are preserved. Tests cover both uncertain recovery and strong-label preservation.
- Browser stabilization has independent Left/Right state, clears current stable labels during candidate transitions and missing observations, and restarts confirmation after gaps. Missing-hand release and reset can target one hand or both.
- Coaching progress resets when its target changes. Missing/invalid centroid references produce unavailable/null alignment feedback instead of claiming every finger is aligned.
- SDK metadata supports the exporter's `hidden_sizes` field and the optional legacy `hidden_dim` field.

Integration details and commands are in [the benchmark guide](../../docs/signer-benchmark.md). Browser consumers must handle nullable `meanError` and `isAligned` and the `referenceAvailable` flag. Existing packaged executables and browser deployments are not rebuilt or published by these source changes.

## Evaluation readiness

Added `python -m ml.benchmark_readiness` to prepare a non-overwriting identity plan and audit signer/source/session separation, canonical duplicate leakage, all-letter/hand coverage, verified identity evidence, and minimum signer coverage.

The actual audit of the current dataset finds **1,648 sessions without verified signer identities**, so it correctly reports **benchmark_ready: false**. See [readiness.json](../signer-benchmark/readiness.json) and [identity-plan.csv](../signer-benchmark/identity-plan.csv). Sessions are already anonymously identifiable; their IDs cannot establish that different sessions were performed by different people. Unknown identities remain blank rather than being counted as independent signers.

Anonymous participant IDs are supported when the same real person's recordings can be grouped consistently. The guide describes prospective supervised collection without storing real names. A genuine signer-separated evaluation requires those recordings or recoverable grouping evidence. Whole-model fine-tuning is therefore not run under a false independence claim; retained checkpoints remain in place.

## Validation

**122 tests passed, 7 export-related warnings.** This includes fresh-frame consumption, duplicate notifications, stalled capture errors, confidence-aware handedness recovery, confirmation after gaps, actual SDK state/coaching replay, unknown-identity refusal, verified anonymous signer acceptance and cross-split leakage rejection.

The existing Windows sandbox test runner changes only temporary-directory creation mode, not application logic or existing ACLs. [Full test log](fixes-tests.log). The SDK was executed through Node's TypeScript stripping; a full browser deployment and TypeScript compiler build were not run. Live human/camera accuracy has not been measured by these synthetic regressions.

Dataset and protected checkpoint hashes were verified unchanged against the prior experiment. No claim of higher model F1 is made from source-only fixes.
