# Jetson Orin 程式實作專案 - Jetpack

這個學習資源旨在讓你快速掌握 **Jetson Orin** 平台的開發流程。你將學會如何設定環境、編譯並執行以 `Eigen` 和 `OpenBLAS` 為基礎的矩陣運算程式，並透過 `tegrastats` 工具即時觀察 CPU、GPU 及記憶體的使用狀況。

## 準備環境

本教材以 Jetson Orin + JetPack 6.2 為例，指令可在大多數 Ubuntu 系統上沿用。若遇到安裝或路徑問題，請依你的環境調整。

安裝指令：
  ```bash
  sudo apt update
  sudo apt install -y build-essential libeigen3-dev libopenblas-dev
  ```
（如已安裝可略過）

## 執行範例

編譯指令（JetPack 6.2 預設路徑）：
```bash
g++ <source_file>.cpp -o <output_binary> -I <eigen_include_path> -L <openblas_lib_path> -lopenblas
```
* `<source_file>.cpp：你的程式碼檔案。
* `<output_binary>：編譯後的執行檔名稱。
* `<eigen_include_path>：Eigen 的標頭檔路徑。
* `<openblas_lib_path>：OpenBLAS 的函式庫路徑。

執行程式：
```bash
./run_demo
```
效能監測（另開終端）：
```bash
sudo tegrastats
```


## Notes

缺乏「學習目標」與「背景說明」
目前的說明偏重「怎麼做」，但沒有解釋「為什麼要做」或「做了能學到什麼」。學生可能只是在執行指令，卻不清楚這些步驟和 Jetson Orin、矩陣運算、效能分析之間的關聯。

操作邏輯不夠明確
例如：為什麼要用 Eigen/OpenBLAS？這兩者有什麼差異？效能監測的意義是什麼？這些操作和日常工程應用有什麼連結？如果能補充「原理簡介」或「應用場景」，學生會更容易理解。

缺乏「預期成果」或「驗收標準」
學生執行完步驟後，應該知道自己是否成功、結果是否合理。例如：執行 tegrastats 應該看到什麼？矩陣運算結果如何判斷正確？這些都可以明確列出。

欠缺「延伸思考」或「挑戰題」
如果只照步驟操作，學習深度有限。可以鼓勵學生思考：如果換成不同的矩陣大小、不同的運算方式，效能有何變化？這樣能培養主動探索的能力。

不同科系學生的需求差異

電機系：可能更關心硬體效能、資源分配、底層運算原理。
工業工程/管理：可能更在意運算效率、流程優化、資源管理。
資訊管理：可能希望了解如何將這些技術應用到資料分析、系統整合。