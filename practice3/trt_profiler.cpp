#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace nvinfer1;
using namespace std;

// Step 1: 建立自訂 profiler 類別，繼承 IProfiler
class MyProfiler : public IProfiler {
public:
    struct Record { string layerName; float timeMs; };
    vector<Record> records;

    void reportLayerTime(const char* layerName, float timeMs) noexcept override {
        records.push_back({layerName, timeMs});
    }

    void print() {
        cout << "==== TensorRT Profiler Report ====" << endl;
        for (auto& r : records) {
            cout << r.layerName << " : " << r.timeMs << " ms" << endl;
        }
    }
};

int main(){
    // Step 2: 建立 builder 與 network
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetworkV2(0U);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

    // Step 3: 載入 ONNX 模型
    if(!parser->parseFromFile("model.onnx", static_cast<int>(ILogger::Severity::kWARNING))){
        cerr << "Failed to parse ONNX model" << endl;
        return -1;
    }

    // Step 4: 建立 config，這裡示範 FP16 精度
    IBuilderConfig* config = builder->createBuilderConfig();
    config->setFlag(BuilderFlag::kFP16);

    // Step 5: 建立 engine 與 context
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    IExecutionContext* context = engine->createExecutionContext();

    // Step 6: 掛上 profiler
    MyProfiler profiler;
    context->setProfiler(&profiler);

    // TODO: 配置 input/output buffer
    // TODO: 將資料拷貝到 GPU
    // TODO: 呼叫 context->enqueueV2(...) 執行推論

    // Step 7: 輸出 profiler 結果
    profiler.print();

    // Step 8: 清理資源
    context->destroy();
    engine->destroy();
    network->destroy();
    parser->destroy();
    builder->destroy();
    config->destroy();
}
