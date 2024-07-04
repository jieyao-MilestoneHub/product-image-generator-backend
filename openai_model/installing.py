import subprocess
import sys
packages = [
    "--upgrade pip",
    "opencv-python",
    "torch torchvision",
    "pillow",
    "numpy",
    "openai==0.28",
    "googletrans==4.0.0-rc1",
    "opencv-python-headless",
    "git+https://github.com/facebookresearch/segment-anything.git",
    "opencv-python pycocotools matplotlib onnxruntime onnx",
    "python-dotenv",
    "git+https://github.com/JiahuiYu/neuralgym",
    "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124",
    "parameter/dlib-19.24.1-cp311-cp311-win_amd64.whl",
    "torch torchvision torchaudio",
    "diffusers",
    "transformers",
    "opencv-python"
]

for package in packages:
    subprocess.run([sys.executable, "-m", "pip", "install", package])
