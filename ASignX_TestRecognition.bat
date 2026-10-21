@echo off
cd /d "%~dp0"
if not exist "ml\models\sign_classifier_canonical.pth" (
    echo Static model missing: ml\models\sign_classifier_canonical.pth
    echo Train the static model before running combined recognition.
    pause
    exit /b 1
)
if not exist "ml\models\sequence_classifier.pth" (
    echo Dynamic model missing: ml\models\sequence_classifier.pth
    echo Train the sequence model before running combined recognition.
    pause
    exit /b 1
)
if exist ".venv-py314\Scripts\python.exe" (
    ".venv-py314\Scripts\python.exe" main.py --sign-model ml/models/sign_classifier_canonical.pth --sequence-model ml/models/sequence_classifier.pth --exposure auto %*
) else (
    python main.py --sign-model ml/models/sign_classifier_canonical.pth --sequence-model ml/models/sequence_classifier.pth --exposure auto %*
)
if errorlevel 1 pause
