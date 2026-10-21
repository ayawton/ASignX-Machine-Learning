# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

mp_data, mp_binaries, mp_imports = collect_all('mediapipe')
a = Analysis(
    ['ml/dynamic_studio_entry.py'],
    pathex=[SPECPATH],
    binaries=mp_binaries,
    datas=mp_data + [('asignx/assets/hand_landmarker.task', 'asignx/assets')],
    hiddenimports=mp_imports,
    excludes=['torch', 'torchvision', 'pytest', 'IPython'],
    hooksconfig={'matplotlib': {'backends': ['Agg']}},
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name='ASignX_DynamicStudio',
    debug=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
