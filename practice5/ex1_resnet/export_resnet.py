import torch
import torchvision

def main():
    print("正在下載並載入 ResNet-50 模型...")
    # 載入預訓練模型
    model = torchvision.models.resnet50(pretrained=True)
    model.eval()
    
    # 若有 GPU 則搬移至 GPU
    if torch.cuda.is_available():
        model = model.cuda()
        dummy_input = torch.randn(1, 3, 244, 244).cuda()
    else:
        dummy_input = torch.randn(1, 3, 244, 244)

    output_file = "resnet50.onnx"
    print(f"正在匯出至 {output_file} ...")
    
    # 匯出 ONNX (使用 opset 17 避免版本衝突)
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        opset_version=17,
        input_names=["input"],
        output_names=["output"]
    )
    print("匯出完成！")

if __name__ == "__main__":
    main()
