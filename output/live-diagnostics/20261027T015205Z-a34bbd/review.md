# Partial candidate live check

User reports completing only a few HELLO attempts, without finishing the planned HELLO block. Keep this run as partial development evidence. Do not count unfinished planned attempts as failures.

The recorded checkpoint SHA256 matches the reviewed-live-HELLO candidate (438615e25dea1bee158f21b38d40cac7ff25cc7b040dd5c3bcdcaa6c9b843691). Recorded marked blocks emitted J five times, Z eight times, HELLO three times, and no signs during REST. There were also six J outputs before any block was marked; their intended movement is unknown and these are excluded from claims about J/REST performance.

HELLO block: three accepted HELLO outputs, two BACKGROUND rejections, and one tracking rejection with 40% hand visibility. Without content review, rejected captures cannot be counted as missed complete salutes. No recorded camera interval exceeded 200 ms. Overall throughput about 29.6 FPS. The REST blocks lasted about 38 seconds total with hands visible about 89% of frames and no emitted signs.

Classification replay and baseline comparison are in replay.json and partial_comparison.json. This is a small same-person partial check, not recognition accuracy or grounds to replace the existing active model. No source labels, model weights, thresholds, or training membership were changed. The next camera check can focus on HELLO salutes and visible-hand rest/repositioning; repeating J/Z immediately is unnecessary.
