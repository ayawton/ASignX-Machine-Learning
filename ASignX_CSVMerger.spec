# -*- mode: python ; coding: utf-8 -*-
a = Analysis(
    ['ml/csv_merger_gui.py'], pathex=[], binaries=[], datas=[],
    hiddenimports=[], hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=['torch', 'torchvision', 'pytest', 'cv2', 'mediapipe', 'matplotlib', 'pandas', 'scipy'],
    noarchive=False, optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='ASignX_CSVMerger',
          debug=False, strip=False, upx=True, console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=True, name='ASignX_CSVMerger')
