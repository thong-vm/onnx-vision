import React, { useState, useRef } from "react";
import * as ort from "onnxruntime-web";
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
    const resizedCtx = resizedCanvas.getContext("2d");

    resizedCtx.drawImage(img, 0, 0, 240, 240);
    const modelResolution = [256, 256];
    const imageData = resizedCtx.getImageData(
      0,
      0,
      modelResolution[0],
      modelResolution[1]
    );

    const inputTensor = await ort.Tensor.fromImage(imageData, {
      resizedWidth: 256,
      resizedHeight: 256,
    });

    const session = await createModelCpu(MODEL_PATH);

    const start = Date.now();
    const feeds = { [session.inputNames[0]]: inputTensor };
    const outputData = await session.run(feeds);
    const end = Date.now();

    setInferenceTime(end - start);
      setOutput(outputData[session.outputNames[0]]);
      alert(JSON.stringify(output, null, 4));
  };

  const createModelCpu = async (url) => {
    return await ort.InferenceSession.create(url, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  };

  return (
    <div className="h-[100vh] w-full">
      <div className="h-1/4">
        <div className="flex justify-end w-full py-5 px-2">
          <span className="text-white">
            Camera info: Camera Dell Laptop 2.0.
          </span>
        </div>
        <div className="flex sm:flex-row flex-col justify-between w-full sm:py-5 sm:px-2">
          <select className="w-[200px] rounded-md focus:outline-none">
            <option value="" key="">
              Test value
            </option>
          </select>
          <div className="flex justify-center">
            <span className="text-4xl font-bold text-white">Kensa Momiji</span>
          </div>
          <div className="flex flex-row space-x-2">
            <button onClick={() => detect("circle_img")}>Detect</button>
            <button onClick={() => detect("circle_img")}>Detect</button>
          </div>
        </div>
      </div>
      <div className="relative w-full h-3/4  sm:-mt-10  py-10 ">
        <div className="h-full overflow-y-auto flex sm:flex-row sm:space-x-2 px-5 sm:px-0 sm:space-y-0 space-y-2 flex-col items-center justify-center ">
          <div className="sm:w-1/2 w-full h-full sm:mt-0 mt-[60%]">
            <img
              ref={imageRef}
              id="circle_img"
              src={TEST_IMAGE}
              alt="Circle"
              className="object-contain h-full w-full rounded-lg"
            />
          </div>
          <div className="sm:w-1/2 w-full h-full ">
            <canvas
              ref={resizedCanvasRef}
              id="resized_ctx"
              width="240"
              height="240"
              className="object-contain h-full w-full"
            ></canvas>
          </div>
        </div>
        {/* <div className="bg-[#D9D9D9] sm:w-[95%] w-full h-full absolute top-0 left-1/2 -translate-x-1/2 opacity-50 z-10 rounded-md"></div> */}
      </div>

      <div id="output_txt">
        {output && (
          <>
            <pre className="hidden">{JSON.stringify(output, null, 4)}</pre>
            <p className="text-white ">Inference time: {inferenceTime} ms</p>
          </>
        )}
      </div>
    </div>
  );
};

export default ModelInference;
