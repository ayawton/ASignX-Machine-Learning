# Extra B after HELLO

The unmarked combined run contains 13 confirmed static B emissions within 0.6 seconds after a HELLO emission. They occur about 0.3 seconds afterward, once dynamic suppression ends. This supports a handoff issue; it does not show that all B handshapes were mislabeled training examples.

The combined test launcher now enables --hold-static-after-gesture. After a dynamic sign is accepted, the moving hand's ending static label is consumed. It cannot immediately be emitted as an extra letter while that pose is held. Another static label stabilizing, or that hand being absent for the configured release frames, rearms it. The HUD shows Release pose during this wait. A brief tracking dropout does not rearm it; an idle second hand is unaffected; rejected dynamic guesses do not consume static labels. For intentional B after HELLO, lower the signing hand out of view for about one second, then form B.

77 focused tests passed. Static and dynamic checkpoint hashes match the recorded originals. No B/HELLO recordings or background labels were changed. A strict thumb-across-palm rule was not added: that is one B form, but relaxed-thumb forms can also be valid in ASL. Reference: https://www.lifeprint.com/asl101/pages-signs/b/b.htm . This fix prevents an extra letter from a consumed dynamic ending pose; it does not retrain the static classifier or establish all-alphabet accuracy.

Next camera check: sign HELLO and hold its ending pose briefly (no extra confirmed B expected), lower the hand out of frame for about one second, then intentionally hold B (B expected). Also check J/Z and a different static letter after a dynamic sign.
