# Background-aware left-hand pilot

**Not ready for promotion:** the candidate rejects the held-out HELLO at the
current confidence threshold and misclassifies the held-out background as HELLO
below threshold. Read review.md before using either experimental launcher.

36 copied recordings: 9 J, 10 Z, 8 HELLO, 9 BACKGROUND. One background take failed tracking and was excluded. Originals and existing checkpoints are unchanged.

Previous positive train/validation/test assignments are preserved. Background takes use seeded take-level 7/1/1 assignments. This remains one signer; it is not a person-independent benchmark. Saved test inference was skipped.

Use Diagnose_Background_Pilot_Sensitive.bat for the same .025 motion-start setting as the last live run; only the dynamic checkpoint changes. Test_Background_Pilot.bat uses the baseline .045 setting. Both write diagnostic reports. Avoid comparing simultaneous changes to model and threshold.
