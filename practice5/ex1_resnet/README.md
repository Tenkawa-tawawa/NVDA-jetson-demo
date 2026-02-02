# 實作範例 (1)：TensorRT 基礎部署流程 (ResNet-50)

## 🎯 實作意義 (為什麼要做這個？)
1.  **掌握標準流程**：AI 模型要落地到邊緣裝置，最標準的路徑就是「PyTorch 訓練 -> 轉 ONNX 中間檔 -> 轉 TensorRT Engine」。這題讓你完整走一遍這個流程。
2.  **效能健檢 (Profiling)**：學會使用 \`trtexec\` 的 \`--dumpProfile\` 功能。這就像醫生的聽診器，能幫你找出模型中哪一層「跑得最慢」，讓你未來知道該如何優化模型結構。

## 🛠️ 事前準備
確保已安裝 PyTorch：
\`\`\`bash
pip install torch torchvision
\`\`\`

## 🚀 實作步驟
1.  **匯出模型**：執行腳本將 PyTorch 轉為 ONNX。
    \`\`\`bash
    python3 export_resnet.py
    \`\`\`
2.  **編譯與分析**：使用 TensorRT 編譯模型並紀錄各層耗時。
    \`\`\`bash
    trtexec --onnx=resnet50.onnx --saveEngine=resnet50_fp32.plan --dumpProfile > resnet50_profile.log
    \`\`\`

## 📊 結果觀察
打開 \`resnet50_profile.log\`，搜尋 **"Profile Results"** 表格。
* **觀察**：找出 \`Time(ms)\` 最大的層。
* **結論**：通常是第一層卷積 (Convolution)，因為輸入圖片解析度最高 (244x244)，計算量最大。
