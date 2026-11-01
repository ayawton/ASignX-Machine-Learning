# Record THANK_YOU, PLEASE and HELP

Open `ASignX_RecordWordSigns.bat` from the project folder. It uses the Python
source recorder with both hands, standard **hands + body** landmarks (180 features),
30 steps at 15 FPS, and a separate `ml/data/sequences_words` output folder.
The launcher requires the project's Python environment; it does not use an older
packaged EXE. Adding labels alone does not update the trained model.

1. Keep your head, upper body and both hands in view. Use the intended signed
   language and a consistent, verified version of each sign.
2. Enter a stable pseudonymous signer ID. Use the same ID for that person across
   labels. Enter a recording ID for each independent recording visit.
3. Choose `THANK_YOU`, open the camera and click **Start take**. After the
   countdown, perform one complete sign and click **Save take**. Start a new take
   for every repetition; discard incomplete or incorrect performances.
4. Close the camera to change the label to `PLEASE`, then `HELP`, and repeat.
   Keep all capture settings and the output folder consistent. The label dropdown
   also allows typed labels.
5. Collect `BACKGROUND` takes of ordinary movements, pauses and confusing
   non-sign movements. Keep valid signs under their proper labels; leave uncertain
   performances out until reviewed.

Aim initially for 30–50 complete takes per new sign spread across several real
recording visits, with varied speed and position. Include additional signers when
possible. This is a collection target, not an accuracy guarantee. Three takes per
label is only a pipeline minimum. Keep some fresh recordings for later evaluation.
Verify that HELP's required hands stay visible throughout: the recorder's quality
threshold only requires some hand visibility, not both hands or a correct sign.
Saved NPZ files contain landmarks, not reviewable video; record video separately
if visual review will be needed.

## Train a separate candidate after recording

Run from the project root:

```powershell
.\.venv-py314\Scripts\python.exe -m ml.train_sequence --data-dir ml/data/sequences_words --output ml/models/sequence_words_candidate.pth --epochs 50
```

This candidate learns only the labels in that dataset. To retain J, Z and HELLO
in one dynamic model, include their reviewed recordings with the same feature
format, hand mode, sequence length and label spelling before training. Do not mix
older 150-feature or hands-only recordings into this folder. Do not relabel or
overwrite the existing recordings to make them fit.

Test the candidate separately:

```powershell
.\ASignX_TestRecognition.bat --sequence-model ml/models/sequence_words_candidate.pth
```

Evaluate complete signs and ordinary movement on fresh recordings before changing
the default launcher. Check confusion between the new words and existing signs,
missed detections and false outputs. This preset does not train a model, establish
recognition accuracy, or enable sentence translation.
