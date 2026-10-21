"""Independent desktop app. Run: python -m ml.csv_merger_gui."""
from __future__ import annotations

from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ml.merge_landmarks import preview_merge, commit_merge


def default_main() -> str:
    roots = [Path.cwd(), Path(__file__).resolve().parent.parent]
    if getattr(sys, 'frozen', False):
        roots.insert(0, Path(sys.executable).resolve().parent)
    for root in roots:
        candidate = root / 'ml' / 'data' / 'landmarks.csv'
        if candidate.is_file():
            return str(candidate.resolve())
    return ''


class MergerApp:
    def __init__(self, root):
        self.root = root
        root.title('ASignX CSV Merger')
        root.geometry('900x690')
        root.minsize(760, 550)
        self.preview = None
        self.busy = False
        self.events = queue.Queue()
        self.main = tk.StringVar(value=default_main())
        self.incoming = tk.StringVar()
        self.status = tk.StringVar(value='Choose both files, then preview. Nothing is saved until you confirm the merge.')
        frame = ttk.Frame(root, padding=20)
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='ASignX CSV Merger', font=('Segoe UI', 20, 'bold')).pack(anchor='w')
        ttk.Label(frame, text='Keep your main dataset and append valid new samples from another device.').pack(anchor='w', pady=(6, 18))
        self.inputs = []
        for title, variable in [('Main CSV (takes priority)', self.main), ('Incoming CSV (from another device)', self.incoming)]:
            ttk.Label(frame, text=title).pack(anchor='w')
            row = ttk.Frame(frame)
            row.pack(fill='x', pady=(4, 12))
            entry = ttk.Entry(row, textvariable=variable)
            entry.pack(side='left', fill='x', expand=True)
            button = ttk.Button(row, text='Browse...', command=lambda v=variable: self.browse(v))
            button.pack(side='right', padx=(8, 0))
            self.inputs.extend([entry, button])
            variable.trace_add('write', self.invalidate)
        ttk.Label(frame, wraplength=710, text='Existing main rows stay unchanged. Incoming exact duplicates, label conflicts and session conflicts are skipped. Source/session IDs are preserved. Similar poses are kept.').pack(anchor='w', pady=(0, 12))
        actions = ttk.Frame(frame)
        actions.pack(fill='x')
        self.preview_button = ttk.Button(actions, text='Preview merge', command=self.start_preview)
        self.preview_button.pack(side='left')
        self.merge_button = ttk.Button(actions, text='Merge into main CSV', command=self.start_merge, state='disabled')
        self.merge_button.pack(side='left', padx=10)
        self.details = tk.Text(frame, height=15, wrap='word', font=('Consolas', 10), state='disabled')
        self.details.pack(fill='both', expand=True, pady=14)
        ttk.Label(frame, textvariable=self.status, wraplength=710).pack(anchor='w')
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(100, self.poll)

    def browse(self, variable):
        path = filedialog.askopenfilename(title='Choose landmarks CSV', filetypes=[('CSV files', '*.csv')])
        if path:
            variable.set(path)

    def invalidate(self, *_):
        self.preview = None
        self.merge_button.configure(state='disabled')
        self.show('File selection changed. Preview again before merging.')

    def show(self, text):
        self.details.configure(state='normal')
        self.details.delete('1.0', 'end')
        self.details.insert('end', text)
        self.details.configure(state='disabled')

    def run(self, kind, action):
        self.busy = True
        for widget in [*self.inputs, self.preview_button, self.merge_button]:
            widget.configure(state='disabled')
        self.status.set('Checking CSV files...' if kind == 'preview' else 'Creating backup and saving merge...')
        def worker():
            try:
                self.events.put((kind, action(), None))
            except Exception as exc:
                self.events.put((kind, None, str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def start_preview(self):
        if not self.main.get().strip() or not self.incoming.get().strip():
            messagebox.showerror('Choose files', 'Select both the main and incoming CSV.')
            return
        main, incoming = self.main.get(), self.incoming.get()
        self.preview = None
        self.run('preview', lambda: preview_merge(main, incoming))

    def start_merge(self):
        snapshot = self.preview
        if snapshot is None:
            return
        count = len(snapshot.accepted)
        if not messagebox.askyesno('Confirm merge', f'Append {count:,} rows into:\n{snapshot.main}\n\nA timestamped backup and detailed report will be saved beside it.\nContinue?'):
            return
        self.run('merge', lambda: commit_merge(snapshot))

    def poll(self):
        try:
            kind, result, error = self.events.get_nowait()
        except queue.Empty:
            self.root.after(100, self.poll)
            return
        self.busy = False
        for widget in [*self.inputs, self.preview_button]:
            widget.configure(state='normal')
        if error:
            self.preview = None
            self.status.set('Operation stopped. Review the error and preview again.')
            self.show(error)
            messagebox.showerror('Merge stopped', error)
        elif kind == 'preview':
            self.preview = result
            counts = result.counts
            lines = [f'Main: {result.main}', f'Incoming: {result.incoming}', '',
                     f'Existing main rows: {result.main_rows:,}', f'Incoming rows: {len(result.decisions):,}',
                     f'New rows to append: {len(result.accepted):,}']
            lines += [f'{name.capitalize()}: {counts.get(name, 0):,}' for name in ('duplicate', 'invalid', 'label conflict', 'session conflict')]
            lines += [f'Rows after merge: {result.main_rows + len(result.accepted):,}', '', 'Skipped rows (first 50; full details saved in merge report):']
            skipped = [d for d in result.decisions if d['decision'] != 'append']
            lines += [f"  Row {d['row']}: {d['decision']} {d.get('reason', '')}" for d in skipped[:50]]
            self.show('\n'.join(lines))
            self.status.set('Preview ready. Check the totals before merging.' if result.accepted else 'No new rows to append. Main CSV needs no changes.')
            self.merge_button.configure(state='normal' if result.accepted else 'disabled')
        else:
            self.preview = None
            backup, report = result
            self.show(f'Merge completed.\n\nBackup:\n{backup}\n\nDetailed report:\n{report}\n\nIncoming file was left untouched.')
            self.status.set('Merge completed. Preview again to check another incoming file.')
        self.root.after(100, self.poll)

    def close(self):
        if self.busy:
            messagebox.showinfo('Operation running', 'Wait for the current operation to finish before closing.')
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    MergerApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
