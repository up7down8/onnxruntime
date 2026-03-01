import ../src/onnxruntime
import os

let testDir = currentSourcePath().parentDir
let modelPath = testDir / "testdata" / "URL-TITLE-classifier" / "model.onnx"

# Test with CoreML
echo "Testing with CoreML..."
let model = loadModel(modelPath, useCoreML=true)
echo "Model loaded successfully with CoreML!"
model.close()
