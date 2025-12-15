#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace nvinfer1;
using namespace std;

// Step 1: 建立 profiler 類別
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
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetworkV2(0U);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

    parser->parseFromFile("model.onnx", static_cast<int>(ILogger::Severity::kWARNING));

    IBuilderConfig* config = builder->createBuilderConfig();
    config->setFlag(BuilderFlag::kFP16);

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    IExecutionContext* context = engine->createExecutionContext();

    // Step 2: 掛上 profiler
    MyProfiler profiler;
    context->setProfiler(&profiler);

    // TODO: 配置 input/output buffer
    // TODO: 將資料拷貝到 GPU
    // TODO: 呼叫 context->enqueueV2(bindings, 0, nullptr) 執行推論

    // Step 3: 輸出 profiler 結果
    profiler.print();

    // Step 4: 清理資源
    context->destroy();
    engine->destroy();
    network->destroy();
    parser->destroy();
    builder->destroy();
    config->destroy();
}
