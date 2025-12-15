#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <iostream>
using namespace nvinfer1;
using namespace std;

int main(){
    // Step 1: 建立 builder 與 network
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Step 2: 建立 parser，載入 ONNX 模型
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    if(!parser->parseFromFile("model.onnx", static_cast<int>(ILogger::Severity::kWARNING))){
        cerr << "Failed to parse ONNX model" << endl;
        return -1;
    }

    // Step 3: 建立 config 與 engine
    IBuilderConfig* config = builder->createBuilderConfig();
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    // Step 4: 建立 context
    IExecutionContext* context = engine->createExecutionContext();

    cout << "Engine built successfully (default FP32 on GPU)" << endl;

    // TODO: 配置 input/output buffer
    // TODO: 執行一次推論，確認 engine 可以跑起來

    // Step 5: 清理資源
    context->destroy();
    engine->destroy();
    network->destroy();
    parser->destroy();
    builder->destroy();
    config->destroy();
}
