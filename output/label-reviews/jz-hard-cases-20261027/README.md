# J/Z training readiness review

Run ASignX_ReviewJZTraining.bat. Nineteen saved captures were selected from the latest combined session: accepted J/Z candidates and J/Z handshape rejections. Selection does not establish actual sign correctness. Scores and model decisions are hidden in the reviewer. All captures retain their original files and hashes in sources.json.

J = recognizable J; Z = recognizable Z; H = recognizable HELLO; B = clear ordinary movement that is not a supported dynamic sign; U = unclear, insufficient geometry, or possibly another sign. Judge the full movement and handshape. Do not infer truth from whether a geometry rule accepted/rejected it. Do not force short valid signs into BACKGROUND. Physical contact is not visible in tracked geometry.

Decisions save to label_review.json in this folder. There are no default answers. This review does not change raw recordings, checkpoint weights, earlier reviews, or training membership. UNSURE examples stay excluded. Reviewed positive and negative examples from this one visit must not be randomly split as if they were independent people or fresh evaluation. If we use them for training, a new recording session is needed to evaluate the candidate.

Training readiness: code and training pipeline are ready, but there are currently no confirmed new J/Z wrong-handshape negatives. The two earlier pointing captures are both UNSURE. Use this review to identify useful evidence first; if the clips are unclear or insufficiently varied, collect paired valid and incorrect-form examples in a new development session.
