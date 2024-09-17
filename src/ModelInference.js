import React, { useState, useRef } from 'react';
import * as ort from 'onnxruntime-web';
//public/
const MODEL_PATH = "/best.onnx";
const TEST_IMAGE = "/100.jpg";

const ModelInference = () => {
  const [output, setOutput] = useState(null);
  const [inferenceTime, setInferenceTime] = useState(null);
  const imageRef = useRef(null);
  const resizedCanvasRef = useRef(null);

  const preprocess = (data, width, height) => {
    // Implement the preprocessing function here
    // This function should match the preprocessing logic in your original code
  };

  const detect = async (imdid) => {
    const img = imageRef.current;
    const resizedCanvas = resizedCanvasRef.current;
    const resizedCtx = resizedCanvas.getContext('2d');

    resizedCtx.drawImage(img, 0, 0, 240, 240);
    const modelResolution = [256, 256];
    const imageData = resizedCtx.getImageData(0, 0, modelResolution[0], modelResolution[1]);

    const inputTensor = await ort.Tensor.fromImage(
      imageData,
      { resizedWidth: 256, resizedHeight: 256 }
    );

    const session = await createModelCpu(MODEL_PATH);
    
    const start = Date.now();
    const feeds = { [session.inputNames[0]]: inputTensor };
    const outputData = await session.run(feeds);
    const end = Date.now();

    setInferenceTime(end - start);
    setOutput(outputData[session.outputNames[0]]);
  };

  const createModelCpu = async (url) => {
    return await ort.InferenceSession.create(url, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
  };

  return (
    <div>
      <img ref={imageRef} id="circle_img" src={TEST_IMAGE} alt="Circle" style={{ display: 'none' }} />
      <canvas ref={resizedCanvasRef} id="resized_ctx" width="256" height="256"></canvas>
      <button onClick={() => detect('circle_img')}>Detect</button>
      <div id="output_txt">
        {output && (
          <>
            <pre>{JSON.stringify(output, null, 4)}</pre>
            <p>Inference time: {inferenceTime} ms</p>
          </>
        )}
      </div>
    </div>
  );
};

export default ModelInference;
