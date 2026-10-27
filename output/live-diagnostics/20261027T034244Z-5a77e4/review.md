# Combined run and pilot form checks

All test blocks were UNMARKED. The log can show emitted outputs but cannot establish which letter or word was intended. Static raw candidates and confirmations are separate; this run alone does not establish all-alphabet accuracy.

User reports movement-driven false J from a closed fist and Z without an extended index finger. User now requests the forehead-salute-only HELLO form, superseding the earlier wave-inclusive pilot policy. Existing model weights and source labels were preserved.

Added opt-in necessary form checks in ml/gesture_forms.py and enabled them in the combined and reviewed-HELLO test launchers. J needs sustained extended-pinky evidence, Z needs sustained extended-index evidence, and both need other long fingers predominantly folded. Where original hand-motion observations are available, an idle second hand cannot supply the moving hand's shape. HELLO needs an open hand starting near a temple proxy; two consecutive start observations are sufficient, without a forced contact hold. The model still handles movement classification. These checks are not full ASL trajectory verification, nor proof of physical head contact. An ordinary wave away from the forehead is outside the narrowed pilot form.

71 focused automated checks passed, including rejecting closed-fist motion despite high class confidence, correct poses under rotation/scale/reflection, idle-other-hand rejection, late head approach, missing face reference, and raw motion trace/replay. Training-only form audit retained 35 HELLO, 7 J and 6 Z training inputs. Form-only replay retained all five inputs that produced HELLO in the previous five-salute session.

The latest combined run's feature-only comparison rejected 1 of 7 recorded J outputs, 7 of 11 Z outputs, and 11 of 19 HELLO outputs. These are proposed necessary-evidence filters on unannotated data, not measured accuracy gains. Original raw motion inputs were not stored in this old run, so this comparison does not reproduce the new moving-hand restriction. A fresh live combined test is required.

Next test: genuine J/Z/salute signs followed by moving a closed fist, moving a non-pointing hand through a Z-like path, and an open-hand movement away from the forehead. Negative movements must not emit J/Z/HELLO. A confirmed static alphabet letter from a held pose can still be expected in combined mode; this is distinct from a dynamic false output. Inspect logs and review uncertain movements before adding training examples.
