#!/bin/bash
export PATH=/usr/src/tensorrt/bin:$PATH

echo "=== 步驟 1: 匯出 ONNX 模型 ==="
python3 export_resnet.py

echo "=== 步驟 2: 編譯 Engine 並進行 Profiling ==="
trtexec --onnx=resnet50.onnx \
        --saveEngine=resnet50_fp32.plan \
        --dumpProfile > resnet50_profile.log

echo "執行完成！請查看 resnet50_profile.log"
