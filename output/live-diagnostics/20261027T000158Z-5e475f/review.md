# Expanded training-v2 live check

The saved report confirms use of sequence_training_v2.pth with the same .025
motion-start threshold, .35-palm-width minimum excursion, .12-second minimum
motion and .8 class-confidence cutoff. This run lasted longer than previous
runs and contains many more performances; compare behavior, not raw counts as
if trial sizes were equal.

| Intended section | Accepted dynamic outputs | Additional decisions |
| --- | --- | --- |
| J | 24 J | 4 BACKGROUND; 1 tracking rejection |
| Z | 25 Z | 1 below-confidence result; 1 insufficient-motion rejection |
| HELLO | 17 HELLO | 14 below-confidence results; 16 BACKGROUND; 2 insufficient-motion rejections |
| REST | none | No tracked hand in any REST frame |

No accepted cross-label outputs occur inside the J, Z or HELLO marked blocks.
One Z output before a target was marked has unknown intended meaning. No
individual attempt annotations or raw video were recorded, so correct/missed
sign accuracy and whether BACKGROUND decisions were reset movements cannot be
established from these logs. Processing averaged 19.10 FPS overall. HELLO hand
visibility was 89.8%; shoulders and face references were present throughout.

The REST block's lack of tracked hands does not test rejection of visible idle
poses, partial movements or return-to-rest paths. Earlier prepared-clip results
still include 4 negative clips incorrectly accepted as HELLO. Retain these
limitations before promoting this candidate.

## Static alphabet labels observed by the user

The diagnostic launcher also loaded the separate static alphabet checkpoint.
Source inspection found that static confirmation is already suppressed while a
dynamic capture is active, but the HUD still displayed the underlying static
guess and its confidence. The earlier logger did not record static candidates
or emissions, so the saved logs cannot prove which observed letters were
actually confirmed.

The HUD now displays Static paused for suppressed alphabet predictions. New
diagnostic logs distinguish static candidates, suppressed status and emitted
letters. The current ASignX_DiagnoseTrainingV2.bat now loads only the dynamic
checkpoint, because J, Z and HELLO are the targets of this diagnostic. Existing
combined-recognition launchers retain static alphabet support. 28 automated
tests passed across the display, logging, gesture transitions and model grouping.

Do not label a correct alphabet pose as static BACKGROUND solely because that
pose appears during a word. The same pose can be intentional fingerspelling or
part of a word. These intents need recognition-mode/context handling. Future
static negatives should be reviewed non-letter examples, separate from the
dynamic model's BACKGROUND dataset.
