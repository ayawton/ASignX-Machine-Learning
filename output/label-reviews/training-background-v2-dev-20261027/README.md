# Training/development BACKGROUND annotation review

This queue contains copies of the 20 BACKGROUND training and 4 BACKGROUND
validation takes in the current training-v2 dataset. The reserved test takes
and separate 20-example evaluation collection are excluded. Model scores are
not used to select or display these examples. Source files are unchanged;
review_sources.json records the original paths, partitions and hashes.

Open the project-root ASignX_ReviewTrainingBackground.bat. Decide based on the
motion, not the old BACKGROUND label:

- HELLO: recognizable salute or greeting wave, including a short valid greeting.
- BACKGROUND: clearly unrelated/non-greeting movement or stationary pose.
- UNSURE: cannot tell, could be another supported sign, or geometry is insufficient.

Do not use "stopped halfway" or absent head contact as automatic negative rules.
Choices save to label_review.json here. They do not alter source takes, existing
checkpoints or prior evaluation annotations. Before retraining, a new snapshot
must apply reviewed labels, exclude UNSURE, verify all classes/partitions and
retain this review's provenance. A user-reviewed stick figure is not original
video or an independent ASL assessment.
