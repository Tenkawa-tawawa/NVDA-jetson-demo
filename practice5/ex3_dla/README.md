# 實作範例 (3)：DLA 硬體架構驗證

## 🎯 實作意義 (為什麼要做這個？)
1.  **認識硬體限制**：並非每張 Jetson 開發板都一樣。高階版 (Orin NX/AGX) 有額外的 **DLA (Deep Learning Accelerator)**，但入門款 (Orin Nano) **只有 GPU**。
2.  **錯誤排查**：學會看錯誤訊息。當軟體試圖呼叫不存在的硬體時，系統會如何回應？這能幫助你未來除錯。
3.  **理解 Fallback**：即使有 DLA，不支援的層也會自動退回 (Fallback) 給 GPU 跑。

## 🛠️ 事前準備
使用範例 (2) 產生的 \`yolov8n.onnx\`。

## 🚀 實作步驟
嘗試強迫使用 DLA 核心 0：
\`\`\`bash
trtexec --onnx=../ex2_yolo/yolov8n.onnx --int8 --useDLACore=0 --allowGPUFallback --dumpLayerInfo
\`\`\`

## 📊 結果觀察
* **預期結果**：執行失敗，出現紅色 Error。
* **錯誤訊息**：\`[E] Cannot create DLA engine, 0 not available\`。
* **結論**：證明這台 Orin Nano 不具備 DLA 硬體。
