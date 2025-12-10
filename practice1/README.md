# Jetson Orin 程式實作專案 - (1)

這份教學整合了從環境確認、程式骨架、效能監測到練習題的完整流程。  
學生可以依照步驟操作，並在最後完成延伸練習題，確認自己真的掌握了內容。

---

## 1. 編譯程式

請依照以下格式編譯程式：

### Eigen 範例
```bash
g++ -O3 -I <eigen_include_path> <source_file>.cpp -o <output_binary>
./<output_binary>
```
