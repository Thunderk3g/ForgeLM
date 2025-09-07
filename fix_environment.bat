
@echo off
echo Fixing LLMForge Environment Issues...

echo.
echo 1. Uninstalling conflicting TensorFlow...
pip uninstall tensorflow tensorflow-gpu -y

echo.
echo 2. Installing CPU-only dependencies...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers[torch] tokenizers safetensors accelerate

echo.
echo 3. Testing installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); import transformers; print(f'Transformers: {transformers.__version__}'); print('Installation successful!')"

echo.
echo Environment fix completed!
pause
