"""Frozen-app entry point with startup diagnostics available before heavy imports."""
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
    diagnostic = None
    trace = None
    if "--self-test" in sys.argv:
        diagnostic = Path(sys.argv[sys.argv.index("--self-test") + 1])
        diagnostic.write_text(json.dumps({"ok": False, "stage": "Loading bundled modules"}), encoding="utf-8")
        import faulthandler
        trace = diagnostic.with_suffix(".trace.txt").open("w", encoding="utf-8")
        faulthandler.dump_traceback_later(20, file=trace)
    try:
        from ml.dynamic_studio_gui import main as studio_main
        studio_main()
    except Exception:
        if diagnostic:
            diagnostic.write_text(json.dumps({"ok": False, "error": traceback.format_exc()}), encoding="utf-8")
        raise
    finally:
        if trace:
            faulthandler.cancel_dump_traceback_later()
            trace.close()


if __name__ == "__main__":
    main()
