#!/bin/bash
cd `dirname $0`
pt_file=./model/det.pt
real_pt_file=$(realpath "$pt_file")
mkdir -p exported_model
exported_model_path=$(pwd)/exported_model
echo $exported_model_path

# Install all required Python dependencies
echo "Installing Python dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python numpy pillow pyyaml tqdm
pip install pycuda
pip install scipy matplotlib pandas seaborn
pip install psutil py-cpuinfo
pip install sympy statsmodels
pip install pynvml

# --- NOVO: Encontra e instala o módulo Python do TensorRT ---
echo "Installing TensorRT Python bindings..."
TENSORRT_WHL=$(find /usr/local/ -name "tensorrt-*-cp*-cp*-linux_x86_64.whl" | head -n 1)
if [[ -f "$TENSORRT_WHL" ]]; then
    python3 -m pip install "$TENSORRT_WHL"
    echo "TensorRT .whl installed from $TENSORRT_WHL"
else
    echo "❌ TensorRT .whl file not found. Please check your TensorRT installation."
fi

# Set up TensorRT Python paths
echo "Setting up TensorRT Python environment..."
export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
export TENSORRT_HOME="/usr/local"

echo "Verifying installations..."
python3 -c "
try:
    import torch
    print('✅ PyTorch:', torch.__version__)
except ImportError as e:
    print('❌ PyTorch not found:', e)

try:
    import tensorrt as trt
    print('✅ TensorRT:', trt.__version__)
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    print('✅ TensorRT builder created successfully!')
except ImportError as e:
    print('❌ TensorRT not found:', e)
except Exception as e:
    print('❌ TensorRT error:', e)

try:
    import ultralytics
    print('✅ Ultralytics installed')
except ImportError as e:
    print('❌ Ultralytics not found:', e)

try:
    import cv2
    print('✅ OpenCV:', cv2.__version__)
except ImportError as e:
    print('❌ OpenCV not found:', e)

try:
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
    print(f'✅ GPU Compute Capability: {major}.{minor}')
except Exception as e:
    print('❌ GPU info error:', e)
"

echo 'start wts converting'
# Fix gen_wts.py for PyTorch 2.5+ compatibility
echo "Patching gen_wts.py for PyTorch 2.5+ compatibility..."
sed -i.backup "s/model = torch.load(pt_file, map_location=device)\['model'\].float()/model = torch.load(pt_file, map_location=device, weights_only=False)['model'].float()/g" model/gen_wts.py

# convert pt file to wts script
python3 model/gen_wts.py -w $pt_file -o $exported_model_path/model.wts -t detect
echo 'finish wts converting'

echo 'Building engine...'
source ../../install/setup.bash
# --- CORRIGIDO: Usando o argumento 's' para o tamanho do modelo ---
ros2 run vision yolov8_det -s $exported_model_path/model.wts /root/booster_ws/robocup_demo/src/vision/model/model.engine s