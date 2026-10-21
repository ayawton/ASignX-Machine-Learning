"""Plain-language static image importer. Run with python -m ml.extractor_gui."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ml.extract_dataset import (
    CLEANING_PRESETS,
    ExtractionReport,
    validate_existing_csv,
    extract_and_clean_dataset,
    main as cli_main,
)


def inspect_existing_csv(csv_path: Path) -> str:
    """Return a human-readable summary of an existing landmarks.csv file."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return "Target file does not exist yet — a new 67-column landmarks.csv will be created."
    try:
        validate_existing_csv(csv_path)
        counts: Counter[str] = Counter()
        sessions: set[str] = set()
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                lbl = (row.get("label") or "").strip().upper()
                sid = (row.get("session_id") or "").strip()
                if lbl:
                    counts[lbl] += 1
                if sid:
                    sessions.add(sid)
        total = sum(counts.values())
        return (
            f"Existing CSV checked: {total:,} samples across {len(counts)} sign classes "
            f"({len(sessions)} sessions). New data will be safely appended without overwriting."
        )
    except Exception as exc:
        return f"Could not inspect existing CSV: {exc}"


def _resolve_default_csv() -> Path:
    """Locate ml/data/landmarks.csv whether launched from project root, dist/, or anywhere else."""
    search_roots: list[Path] = [Path.cwd()]
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        search_roots.extend([exe_dir, exe_dir.parent, exe_dir.parent.parent])
    else:
        here = Path(__file__).resolve().parent.parent
        search_roots.append(here)

    for base in search_roots:
        candidate = base / "ml" / "data" / "landmarks.csv"
        if candidate.exists() or candidate.parent.exists():
            return candidate.resolve()
    return (Path.cwd() / "landmarks.csv").resolve()


class ASignXExtractorGUI:
    """Desktop GUI for the Standalone ASignX Static Dataset Extractor."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ASignX — Static Sign Dataset Importer")
        self.root.geometry("1000x820")
        self.root.minsize(900, 740)
        self.root.configure(bg="#0f172a")

        default_csv = _resolve_default_csv()

        self.input_path_var = tk.StringVar(value="")
        self.output_csv_var = tk.StringVar(value=str(default_csv))
        self.use_staging_var = tk.BooleanVar(value=False)
        self.staging_dir_var = tk.StringVar(value="")
        self.cleaning_var = tk.StringVar(value=next(iter(CLEANING_PRESETS)))
        self.manifest_var = tk.StringVar(value="")
        self.source_id_var = tk.StringVar(value="")
        self.mad_threshold_var = tk.DoubleVar(value=0.0)
        self.dedup_epsilon_var = tk.DoubleVar(value=0.0)
        self.min_sessions_var = tk.IntVar(value=5)
        self.min_conf_var = tk.DoubleVar(value=0.3)
        self.csv_status_var = tk.StringVar(value=inspect_existing_csv(default_csv))
        self.progress_msg_var = tk.StringVar(value="Ready. Select a static sign dataset folder or .zip archive.")
        self.progress_pct_var = tk.DoubleVar(value=0.0)

        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._is_running = False

        self._build_ui()
        self.output_csv_var.trace_add("write", self._on_csv_path_changed)

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background="#0f172a")
        style.configure("Card.TFrame", background="#1e293b", relief="flat")
        style.configure(
            "Title.TLabel",
            background="#0f172a",
            foreground="#f8fafc",
            font=("Segoe UI", 15, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background="#0f172a",
            foreground="#94a3b8",
            font=("Segoe UI", 9),
        )
        style.configure(
            "Section.TLabel",
            background="#1e293b",
            foreground="#38bdf8",
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "Field.TLabel",
            background="#1e293b",
            foreground="#e2e8f0",
            font=("Segoe UI", 9),
        )
        style.configure(
            "Status.TLabel",
            background="#1e293b",
            foreground="#4ade80",
            font=("Segoe UI", 8, "italic"),
        )
        style.configure(
            "Primary.TButton",
            background="#2563eb",
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padding=(14, 8),
        )
        style.map("Primary.TButton", background=[("active", "#1d4ed8"), ("disabled", "#475569")])
        style.configure(
            "Secondary.TButton",
            background="#334155",
            foreground="#f8fafc",
            font=("Segoe UI", 9),
            padding=(10, 5),
        )
        style.map("Secondary.TButton", background=[("active", "#475569")])

        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)

        # Header
        ttk.Label(
            outer,
            text="ASignX Static Hand-Sign Dataset Extractor",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "Choose images, check their sign names, then add clean hand points to your dataset."
            ),
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 12))

        # Card 1: Dataset Input & Target CSV
        io_card = ttk.Frame(outer, style="Card.TFrame", padding=14)
        io_card.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(io_card, text="1. Dataset Source & Target landmarks.csv", style="Section.TLabel").pack(
            anchor="w", pady=(0, 8)
        )

        # Source row
        src_row = ttk.Frame(io_card, style="Card.TFrame")
        src_row.pack(fill=tk.X, pady=4)
        ttk.Label(src_row, text="Input Dataset:", style="Field.TLabel", width=16).pack(side=tk.LEFT)
        ttk.Entry(src_row, textvariable=self.input_path_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8)
        )
        ttk.Button(
            src_row,
            text="Browse Folder...",
            style="Secondary.TButton",
            command=self._browse_input_folder,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            src_row,
            text="Browse .ZIP...",
            style="Secondary.TButton",
            command=self._browse_input_zip,
        ).pack(side=tk.LEFT)

        # Output CSV row
        dst_row = ttk.Frame(io_card, style="Card.TFrame")
        dst_row.pack(fill=tk.X, pady=4)
        ttk.Label(dst_row, text="Target CSV:", style="Field.TLabel", width=16).pack(side=tk.LEFT)
        ttk.Entry(dst_row, textvariable=self.output_csv_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8)
        )
        ttk.Button(
            dst_row,
            text="Select CSV...",
            style="Secondary.TButton",
            command=self._browse_output_csv,
        ).pack(side=tk.LEFT)

        ttk.Label(io_card, textvariable=self.csv_status_var, style="Status.TLabel").pack(
            anchor="w", pady=(4, 4)
        )

        # Optional Staging Folder row
        stg_row = ttk.Frame(io_card, style="Card.TFrame")
        stg_row.pack(fill=tk.X, pady=(6, 0))
        chk = tk.Checkbutton(
            stg_row,
            text="Save a copy of accepted images in folders named by sign:",
            variable=self.use_staging_var,
            bg="#1e293b",
            fg="#e2e8f0",
            selectcolor="#0f172a",
            activebackground="#1e293b",
            activeforeground="#ffffff",
            font=("Segoe UI", 9),
        )
        chk.pack(side=tk.LEFT)
        ttk.Entry(stg_row, textvariable=self.staging_dir_var, width=28).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=8
        )
        ttk.Button(
            stg_row,
            text="Choose folder...",
            style="Secondary.TButton",
            command=self._browse_staging_dir,
        ).pack(side=tk.LEFT)

        cfg_card = ttk.Frame(outer, style="Card.TFrame", padding=14)
        cfg_card.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(cfg_card, text="2. Choose how to clean new images", style="Section.TLabel").pack(anchor="w")
        ttk.Combobox(cfg_card, textvariable=self.cleaning_var,
                     values=list(CLEANING_PRESETS), state="readonly", width=45).pack(anchor="w", pady=6)
        ttk.Label(cfg_card, text="Always skip unreadable images, unusable hand points and exact repeats. Existing rows stay unchanged.",
                  style="Field.TLabel").pack(anchor="w")
        ttk.Label(cfg_card, text="Unusual poses may be valid examples. Use the last option only after checking your images.",
                  style="Field.TLabel").pack(anchor="w", pady=(2, 8))
        mapping_row = ttk.Frame(cfg_card, style="Card.TFrame")
        mapping_row.pack(fill=tk.X)
        ttk.Label(mapping_row, text="Sign names file (optional):", style="Field.TLabel").pack(side=tk.LEFT)
        ttk.Entry(mapping_row, textvariable=self.manifest_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Button(mapping_row, text="Choose CSV...", command=self._browse_manifest).pack(side=tk.LEFT)
        ttk.Button(mapping_row, text="Preview sign names", command=self._preview_input).pack(side=tk.LEFT, padx=(8, 0))
        source_row = ttk.Frame(cfg_card, style="Card.TFrame")
        source_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(source_row, text="Dataset source name (optional):", style="Field.TLabel").pack(side=tk.LEFT)
        ttk.Entry(source_row, textvariable=self.source_id_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Label(cfg_card, text="Reuse a source name when importing copies of the same dataset. A names file uses: path,label,group.",
                  style="Field.TLabel").pack(anchor="w", pady=(6, 0))
        ttk.Label(cfg_card, text="Group means the original signer or recording. Missing groups stay together; extraction does not invent training splits.",
                  style="Field.TLabel").pack(anchor="w", pady=(2, 0))

        # Action & Progress Bar
        action_frame = ttk.Frame(outer)
        action_frame.pack(fill=tk.X, pady=(2, 8))

        self.run_btn = ttk.Button(
            action_frame,
            text="Extract and add new data",
            style="Primary.TButton",
            command=self._start_extraction,
        )
        self.run_btn.pack(side=tk.LEFT, padx=(0, 14))

        prog_col = ttk.Frame(action_frame)
        prog_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.progress_bar = ttk.Progressbar(
            prog_col, variable=self.progress_pct_var, maximum=100.0, mode="determinate"
        )
        self.progress_bar.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(prog_col, textvariable=self.progress_msg_var, style="Subtitle.TLabel").pack(anchor="w")

        # Card 3: Report Output Console
        log_card = ttk.Frame(outer, style="Card.TFrame", padding=12)
        log_card.pack(fill=tk.BOTH, expand=True)
        ttk.Label(log_card, text="3. Results and skipped images", style="Section.TLabel").pack(
            anchor="w", pady=(0, 6)
        )

        self.report_text = tk.Text(
            log_card,
            bg="#090d16",
            fg="#e2e8f0",
            insertbackground="#ffffff",
            font=("Consolas", 9),
            relief="flat",
            padx=10,
            pady=8,
            height=12,
        )
        self.report_text.pack(fill=tk.BOTH, expand=True)
        self._append_log(
            "Choose a folder or ZIP containing still images. Common folder names are detected automatically.\n"
            "Use Preview sign names before extraction. For unusual layouts, provide a CSV with path,label,group columns.\n"
            "Videos and animated GIFs are not accepted. This model stores one hand per sample.\n"
            "Existing rows are validated and preserved. Each run saves a detailed results file beside the output CSV.\n"
        )

    def _on_csv_path_changed(self, *_: object) -> None:
        raw = self.output_csv_var.get().strip()
        if raw:
            self.csv_status_var.set(inspect_existing_csv(Path(raw)))

    def _browse_manifest(self) -> None:
        path = filedialog.askopenfilename(title="Choose sign names CSV", filetypes=[("CSV", "*.csv")])
        if path:
            self.manifest_var.set(path)

    def _preview_selected_input(self, path_str: str) -> None:
        self.progress_msg_var.set("Dataset selected. Preview sign names to check the folder mapping.")

    def _preview_input(self) -> None:
        if self._is_running:
            return
        path = Path(self.input_path_var.get().strip())
        manifest = self.manifest_var.get().strip()
        if not self.input_path_var.get().strip():
            messagebox.showwarning("Choose images", "Select a folder or ZIP first.")
            return
        self._is_running = True
        self.run_btn.configure(state="disabled")
        def worker():
            import tempfile
            from ml.extract_dataset import discover_static_dataset, load_label_map, safe_extract_zip
            try:
                with tempfile.TemporaryDirectory(prefix="asignx_preview_") as tmp:
                    if path.is_file() and path.suffix.lower() == ".zip":
                        safe_extract_zip(path, Path(tmp))
                        root = Path(tmp)
                    elif path.is_dir():
                        root = path
                    else:
                        raise ValueError("Choose an existing folder or ZIP archive.")
                    found = discover_static_dataset(root, root_label_hint=path.stem,
                                label_map=load_label_map(manifest) if manifest else None)
                    counts = Counter(label for _, label, _, _ in found)
                    groups = {(label, group) for _, label, group, _ in found}
                    msg = "Sign names preview (check these before importing):\n" + "\n".join(
                        f"  {label}: {count} images; {sum(lbl == label for lbl, _ in groups)} source group(s)"
                        for label, count in sorted(counts.items()))
                    msg += "\nIf a sign name is wrong, supply a path,label,group CSV. Never treat random image batches as independent signers."
                    self._queue.put(("preview", msg))
            except Exception as exc:
                self._queue.put(("error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()
        self.root.after(100, self._poll_queue)

    def _browse_input_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select Static Hand-Sign Dataset Folder")
        if folder:
            self.input_path_var.set(folder)
            self._preview_selected_input(folder)

    def _browse_input_zip(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select Static Hand-Sign Dataset .ZIP Archive",
            filetypes=[("ZIP Archives", "*.zip"), ("All Files", "*.*")],
        )
        if file_path:
            self.input_path_var.set(file_path)
            self._preview_selected_input(file_path)

    def _browse_output_csv(self) -> None:
        file_path = filedialog.asksaveasfilename(
            title="Select Target landmarks.csv (Will Append Non-Destructively)",
            defaultextension=".csv",
            initialfile="landmarks.csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if file_path:
            self.output_csv_var.set(file_path)

    def _browse_staging_dir(self) -> None:
        folder = filedialog.askdirectory(title="Select Canonical Accepted Staging Folder")
        if folder:
            self.staging_dir_var.set(folder)
            self.use_staging_var.set(True)

    def _append_log(self, text: str) -> None:
        self.report_text.configure(state="normal")
        self.report_text.insert(tk.END, text + "\n")
        self.report_text.see(tk.END)
        self.report_text.configure(state="disabled")

    def _start_extraction(self) -> None:
        if self._is_running:
            return
        input_str = self.input_path_var.get().strip()
        output_str = self.output_csv_var.get().strip()
        if not input_str:
            messagebox.showwarning("Missing Input", "Please select a dataset folder or .zip archive first.")
            return
        if not output_str:
            messagebox.showwarning("Missing Target CSV", "Please specify a target landmarks.csv path.")
            return

        input_path = Path(input_str)
        if not input_path.exists():
            messagebox.showerror("File Not Found", f"Input dataset path does not exist:\n{input_path}")
            return

        staging_dir: str | None = None
        if self.use_staging_var.get() and self.staging_dir_var.get().strip():
            staging_dir = self.staging_dir_var.get().strip()

        mad_threshold, dedup_epsilon = CLEANING_PRESETS[self.cleaning_var.get()]
        manifest_path = self.manifest_var.get().strip() or None
        source_id = self.source_id_var.get().strip() or None
        self._is_running = True
        self.run_btn.configure(state="disabled")
        self.progress_pct_var.set(0.0)
        self._append_log("-" * 72)
        self._append_log(f"Starting extraction from: {input_path}")
        self._append_log(f"Target CSV (append mode): {output_str}")
        if staging_dir:
            self._append_log(f"Consolidating accepted images into: {staging_dir}")

        worker = threading.Thread(
            target=self._run_worker,
            args=(
                input_path,
                Path(output_str),
                staging_dir,
                mad_threshold, dedup_epsilon, 3, 0.5, manifest_path, source_id,
            ),
            daemon=True,
        )
        worker.start()
        self.root.after(100, self._poll_queue)

    def _run_worker(
        self,
        input_path: Path,
        output_csv: Path,
        staging_dir: str | None,
        mad_threshold: float,
        dedup_epsilon: float,
        min_sessions: int,
        min_conf: float,
        manifest_path: str | None,
        source_id: str | None,
    ) -> None:
        import traceback

        t0 = time.time()

        def on_progress(curr: int, total: int, msg: str) -> None:
            pct = (curr / max(1, total)) * 100.0
            self._queue.put(("progress", (pct, msg)))

        try:
            report = extract_and_clean_dataset(
                input_path=input_path,
                output_csv=output_csv,
                staging_dir=staging_dir,
                mad_z_threshold=mad_threshold,
                duplicate_epsilon=dedup_epsilon,
                min_sessions=min_sessions,
                min_detection_confidence=min_conf,
                progress_callback=on_progress,
                manifest_path=manifest_path, source_id=source_id,
            )
            elapsed = time.time() - t0
            self._queue.put(("done", (report, elapsed, output_csv)))
        except Exception as exc:
            tb = traceback.format_exc()
            self._queue.put(("error", f"{exc}\n{tb}"))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "progress":
                    pct, msg = payload  # type: ignore[misc]
                    self.progress_pct_var.set(float(pct))
                    self.progress_msg_var.set(str(msg))
                elif kind == "preview":
                    self._is_running = False
                    self.run_btn.configure(state="normal")
                    self.progress_msg_var.set("Preview complete. Check the sign names in the results below.")
                    self._append_log(str(payload))
                    return
                elif kind == "done":
                    report, elapsed, output_csv = payload  # type: ignore[misc]
                    self._on_extraction_done(report, elapsed, output_csv)
                    return
                elif kind == "error":
                    self._is_running = False
                    self.run_btn.configure(state="normal")
                    self.progress_msg_var.set("Extraction failed.")
                    self._append_log(f"[ERROR] {payload}")
                    messagebox.showerror("Extraction Error", str(payload))
                    return
        except queue.Empty:
            pass

        if self._is_running:
            self.root.after(100, self._poll_queue)

    def _on_extraction_done(self, report: ExtractionReport, elapsed: float, output_csv: Path) -> None:
        self._is_running = False
        self.run_btn.configure(state="normal")
        self.progress_pct_var.set(100.0)
        self.csv_status_var.set(inspect_existing_csv(output_csv))

        lines = [
            f"Extraction finished in {elapsed:.2f}s!",
            f"  • Total images scanned            : {report.total_files_scanned}",
            f"  • Repeated image files           : {report.exact_duplicates_skipped}",
            f"  • No hand detected by MediaPipe   : {report.no_hand_detected}",
            f"  • Unusable hand shapes           : {report.anatomical_outliers_rejected}",
            f"  • Unusual poses (optional)       : {report.statistical_outliers_rejected}",
            f"  • Repeated / similar examples    : {report.near_duplicates_pruned}",
            f"  • Already in CSV skipped          : {report.already_in_csv_skipped}",
            f"  • NEW EXAMPLES ADDED              : +{report.samples_appended}",
        ]
        lines.extend([f"  Unreadable images: {report.unreadable_images}",
                      f"  Conflicting sign labels skipped: {report.label_conflicts_skipped}",
                      f"  Images with multiple hands: {report.multiple_hands_skipped}",
                      f"  Animated images skipped: {report.animated_images_skipped}",
                      f"  Detailed results: {report.audit_path}"])
        lines.extend(f"  Note: {warning}" for warning in report.warnings)
        if report.per_label_counts:
            lines.append("  • Per-Sign Breakdown:")
            for label, count in sorted(report.per_label_counts.items()):
                lines.append(f"      [{label}] : +{count} clean samples")
        self._append_log("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    # If CLI arguments like --input are passed, run in headless CLI mode
    if args and any(a.startswith("-") for a in args):
        cli_main(args)
        return

    root = tk.Tk()
    ASignXExtractorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

