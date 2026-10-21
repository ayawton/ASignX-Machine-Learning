# User-reviewed M correction

One M sample was removed at the user's explicit instruction. The original CSV is landmarks.before.csv; its SHA-256 matches the historical canonical checkpoint. Use this snapshot to reproduce old metrics. The current CSV has 159,986 rows. reviewed.splits.json remaps the preserved partitions and is a data-only manifest; no model was retrained or checkpoint metadata changed. removed-samples.csv preserves the removed data, and row-remap.csv translates historical to current CSV lines.
