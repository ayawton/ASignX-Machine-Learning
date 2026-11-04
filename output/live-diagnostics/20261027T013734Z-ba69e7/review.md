# Fresh same-person session review

Signer: pilot_user_001. User reports HELLO salutes only, more than five attempts, no greeting waves. User reports marking REST while repositioning. Logged target transitions are UNMARKED -> J -> REST -> Z -> REST -> HELLO -> REST; repeated same-key presses cannot be reconstructed from these target states.

The existing training-v2 checkpoint and acceptance settings were retained (SHA256 a488a526d9b9ec79dfb3f917b43ea7df62fe7ced4799b88cde17e0e2756606ee).

- J block: 5 accepted J outputs.
- Z block: 5 accepted Z outputs; one additional Z candidate stayed below threshold.
- HELLO block: 3 accepted HELLO outputs; 4 additional HELLO candidates below 0.8 (0.7990, 0.7126, 0.6914, 0.7648); 11 captured segments classified as BACKGROUND.
- REST blocks: no accepted sign outputs; 2 captured segments classified as BACKGROUND.
- Observed throughput about 29.6 FPS; no recorded capture intervals over 200 ms. HELLO block hand, shoulder, and face reference visibility were 100% at logged frame level.

These are output and capture counts, not per-sign accuracy. Captured segments can be return movements, separate portions of one performance, or complete signs. Reviewing the 18 HELLO-block captures is required before assigning missed-positive labels. Current evidence suggests a classification/capture-content issue rather than a general HELLO tracking outage; it does not establish the cause of each miss.

ASignX_ReviewFreshHello.bat reviews all 18 HELLO-block captures with model scores hidden. H means a recognizable HELLO including a short salute; B means ordinary non-greeting repositioning; U means unclear. Review decisions are saved separately in label_review.json, without modifying original traces or training weights. replay.json records classification replay only, not motion-guard replay.

Greeting-wave acceptance remains an intended scope decision. This run did not test waves, and wave recognition has not been validated. No retraining or threshold changes made.
