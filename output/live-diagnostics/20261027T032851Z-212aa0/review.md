# Five-salute candidate follow-up

User reports performing five salutes. Checkpoint SHA256 matches the reviewed-live candidate (438615e25dea1bee158f21b38d40cac7ff25cc7b040dd5c3bcdcaa6c9b843691).

The HELLO-marked block emitted five HELLO outputs and no other signs. HELLO confidence range: 0.9768 to 0.9858. There were five additional captured movements classified BACKGROUND, alternating with the HELLO outputs. The movement content of these rejects has not been human-reviewed; do not automatically count them as missed salutes or confirmed correct negatives.

REST lasted about 6.2 seconds with hand visibility 100%, and produced no sign outputs. There were no completed motion captures in this block, so it does not establish rejection of varied moving negatives. The requested 15-second movement check was not fully represented in this saved REST block. The earlier partial candidate run had about 38 seconds across REST blocks with no outputs, but limited captured movements.

Processing averaged about 15.7 FPS overall and 14.9 FPS during HELLO, lower than the earlier approximately 29.6 FPS run. No capture interval exceeded 200 ms, and no tracking/motion-evidence rejections were logged. These logs do not establish why processing was slower.

Five outputs match the five user-reported salutes. This is encouraging same-person pilot evidence; no per-sign correctness or general accuracy metric is established without reviewed alignment. The run was not used for training and no model/threshold/launcher changes were made. The known older background false HELLO remains unresolved; a short visible-hand non-greeting movement check is the next priority before broader use.

Replay data is in replay.json. Replay checks classification from captured landmark inputs, not automatic motion boundaries or ASL correctness.
