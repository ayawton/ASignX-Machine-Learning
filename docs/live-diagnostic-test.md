# J / Z / HELLO live diagnostic

## Combined alphabet and reviewed HELLO check

The combined launcher also enables `--hold-static-after-gesture`. After an
accepted dynamic sign, its moving hand's ending static label is consumed by that
gesture. A held HELLO ending pose must not immediately emit an extra B. The HUD
shows Release pose until another static letter stabilizes or that hand leaves
view for the configured release frames (10 by default). A brief tracking dropout
does not release it. For an intentional same-shape letter immediately after a
gesture, lower the signing hand out of view for about one second first. An idle
second hand is not consumed. Rejected/uncertain dynamic guesses do not consume
static letters. This is an isolated-sign interaction rule, not a thumb-shape
restriction; relaxed-thumb B variants can also be valid ASL. Original static
training labels and weights are unchanged.

The current combined test enables `--gesture-form-checks`: J must have sustained
extended-pinky evidence with folded other fingers, Z must have sustained extended
index evidence with folded other fingers, and HELLO must begin open-handed near
the forehead. This follows the user's latest salute-only pilot request. Earlier
wave-inclusive sessions remain historical evidence; their labels are unchanged.
These checks are preliminary geometry requirements, not full ASL trajectory
verification or physical-contact detection. The form policy version is logged;
new optional input traces also retain original hand-motion observations so the
moving-hand restriction can be replayed. Old traces without these observations
can support feature-only comparisons, not exact replay of that restriction.

`ASignX_TestAllSigns.bat` loads the existing 24-class static alphabet checkpoint
and the experimental reviewed-live HELLO/J/Z/BACKGROUND checkpoint together.
This covers all 26 alphabet letters plus HELLO; other word signs have not been
added. Use the left hand initially. Before pressing a diagnostic target key,
test A through I and K through Y, holding each for about two seconds and relaxing
for one second between letters. Static results are logged under UNMARKED; raw
candidate guesses and confirmed static emissions are separate. The letter you
intend to sign is not automatically annotated, so these logs alone cannot give
per-letter accuracy. Note missed or wrong letters while testing.

Then use 7 for five J signs, 8 for five Z signs, 9 for five HELLO salutes, and
0 for 15-20 seconds of ordinary rest/repositioning. Q saves and exits. Static
confirmation pauses during movement/settling. This run checks interaction of
the static and dynamic routes, including unwanted dynamic outputs while forming
letters. Each report records both checkpoint hashes and acceptance settings.
The candidate remains experimental; previous launchers retain their models.

For the fresh same-person check, launch `ASignX_DiagnoseFreshSession.bat`.
It keeps the current training-v2 checkpoint and settings, identifies the signer
as the existing `pilot_user_001`, and creates a new uniquely named session folder.
Use the left hand and slightly change camera distance or position while keeping
face, shoulders, and the signing hand visible. Press 7 for five J signs, 8 for
five Z signs, and 9 for five HELLO salutes. Greeting waves remain an intended
HELLO form, but their training coverage and recognition need separate validation. Pause
one second between signs and two seconds before switching blocks. Press 0 for
20 seconds of held poses and ordinary repositioning, avoiding greeting waves in
this block, then Q to save. This checks fresh-session behavior with the same
person; it does not measure performance on other people or add training data.

Close other camera windows, then launch `ASignX_DiagnosePilot.bat` from the project
root. It uses the existing left-hand pilot checkpoint and current acceptance
thresholds. It does not train a model or save camera images/landmark coordinates.

With the camera window focused:

1. Press 7. Sign J five times naturally with a one-second pause after each take.
2. Pause two seconds; press 8. Do the same for Z.
3. Pause two seconds; press 9. Do the same for HELLO.
4. Pause two seconds; press 0. Rest, then reposition normally for about ten seconds.
5. Press Q. The console prints the saved report location.

Keep face, shoulders and the signing hand visible. Do not exaggerate movements
just to pass a threshold. Use the same signing hand as the pilot (left).

Reports are saved in a unique folder under `output/live-diagnostics/`:

- `frames.jsonl`: per-frame timing, visibility flags, manually selected test block,
  movement state, predictions and rejection events.
- `summary.json`: observed processing FPS, detector/recognizer timing, capture
  intervals, tracking coverage and emissions/rejections by test block.

The keys label your intended test block; they are not proof that the sign was
performed correctly. A pause before changing labels reduces events spilling
between blocks. Whole-block visibility includes rest and does not equal the
gesture-specific sampled visibility shown in a rejection. Timing can reveal
processing delays, but cannot alone establish camera blur. No automated accuracy
claim or threshold change follows just from generating these logs.

After the run, inspect whether failures follow low hand visibility, long capture
intervals, insufficient motion, or confident wrong labels. Then adjust tracking,
capture, or validation-fitted thresholds as the evidence warrants. Preserve
positive and negative cases rather than tuning to successful positives alone.

The current training-v2 launcher, ASignX_DiagnoseTrainingV2.bat, tests dynamic
J/Z/HELLO without loading the static alphabet classifier. Combined-recognition
launchers remain available. In combined mode, Static paused means alphabet
confirmation is suppressed during dynamic capture; a raw letter guess is not a
confirmed letter. New logs separate static_candidates from static_emitted.
For rejection testing in the REST block, include ordinary repositioning and held
poses with the hand visible; no-hand idle periods alone are not enough.

For exact input replay, the current training-v2 launcher enables
--capture-gesture-inputs. It saves completed landmark sequences and original
timestamps in gesture_inputs/ beside the normal logs; no camera video is saved.
Without this flag, diagnostics retain only timing/visibility/prediction metrics.
Captured inputs are diagnostic artifacts, not automatically reviewed training
examples. A replay checks preparation and classification; it does not prove
the automatic capture boundary or ASL correctness.

The 0 key only marks a REST test block; it does not disable recognition or make
every movement in that block a valid negative example. HELLO accepts both the
forehead-salute form and ordinary greeting waves. A genuine greeting wave can
correctly produce HELLO even in a block marked REST. Review movement content
before counting false detections; absent head contact alone is not evidence.

Completed body-context takes use timestamp interpolation at the checkpoint's
sampling rate. Ordinary camera intervals below that rate do not become synthetic
missing-hand observations. Real missing detections and excessive camera gaps
remain subject to tracking checks.
