import React, { useEffect } from "react";
import Webcam from "react-webcam";
import "./App.css";
import { DetectionModel } from "./Model";

function App() {
  const webcamRef = React.useRef(null);
  const canvasRef = React.useRef(null);
  useEffect(() => {
    const detectModel = new DetectionModel();
    detectModel.run(webcamRef, canvasRef);
  }, []);
  return (
    <div className="App">
      <div>
        <Webcam className="webcam" ref={webcamRef} videoConstraints={{
    facingMode: "user", // Ensures it uses the front-facing camera
  }}
  style={{
    transform: "scaleX(-1)",
  }}/>
        <canvas className="canvas" ref={canvasRef}></canvas>
      </div>
    </div>
  );
}

export default App;
