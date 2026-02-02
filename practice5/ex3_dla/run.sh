#!/bin/bash
export PATH=/usr/src/tensorrt/bin:$PATH

echo "=== 警告：硬體檢測 ==="
echo "嘗試執行 DLA 編譯指令..."
echo "預期結果：因為 Orin Nano 沒有 DLA，以下指令應該會報錯 (Error: DLA not supported)。"
echo "=============================================="

trtexec --onnx=../ex2_yolo/yolov8n.onnx \
        --saveEngine=yolov8n_dla_int8.plan \
        --int8 \
        --useDLACore=0 \
        --allowGPUFallback \
        --dumpLayerInfo
