import * as tf from "@tensorflow/tfjs";
import "core-js/stable";

export class DetectionModel {
  private async loadModel() {
    try {
      await tf.ready();
      const modelPath =
        "https://tfhub.dev/tensorflow/tfjs-model/ssd_mobilenet_v2/1/default/1";
      const model = await tf.loadGraphModel(modelPath, { fromTFHub: true });
      return model;
    } catch (e) {
      console.error("Error loading model", e);
    }
  }

  private async detect(model: any, webcamRef: any, canvasRef: any) {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      const video = webcamRef.current.video;
      const ctx = canvasRef.current.getContext("2d");
      const webcamHeight = video.videoHeight;
      const webcamWidth = video.videoWidth;
      const webcamTensor = tf.browser.fromPixels(video);
      console.log("video", webcamHeight, webcamWidth);
      // SSD Mobilenet single batch
      const readyTensor = tf.expandDims(webcamTensor, 0);
      const results = await model.executeAsync(readyTensor);

      // Get a clean tensor of top indices
      const detectionThreshold = 0.4;
      const iouThreshold = 0.5;
      const maxBoxes = 20;
      const prominentDetection = tf.topk(results[0]);
      const justBoxes = results[1].squeeze();
      const justValues = prominentDetection.values.squeeze();

      // Move results back to JavaScript in parallel
      const [maxIndices, scores, boxes] = await Promise.all([
        prominentDetection.indices.data(),
        justValues.array(),
        justBoxes.array(),
      ]);

      // https://arxiv.org/pdf/1704.04503.pdf, use Async to keep visuals
      const nmsDetections = await tf.image.nonMaxSuppressionWithScoreAsync(
        justBoxes, // [numBoxes, 4]
        justValues, // [numBoxes]
        maxBoxes,
        iouThreshold,
        detectionThreshold,
        1 // 0 is normal NMS, 1 is Soft-NMS for overlapping support
      );

      const chosen = await nmsDetections.selectedIndices.data();
      // Mega Clean
      tf.dispose([
        results[0],
        results[1],
        // model, don't clean this one up for loops
        nmsDetections.selectedIndices,
        nmsDetections.selectedScores,
        prominentDetection.indices,
        prominentDetection.values,
        webcamTensor,
        readyTensor,
        justBoxes,
        justValues,
      ]);

      // clear everything each round
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      chosen.forEach((detection) => {
        ctx.strokeStyle = "#00FF00";
        ctx.lineWidth = 2;
        ctx.globalCompositeOperation = "destination-over";
        const detectedIndex = maxIndices[detection];
        const detectedClass = CLASSES[detectedIndex];
        const detectedScore = scores[detection];
        const dBox = boxes[detection];

        // No negative values for start positions
        const startY = dBox[0] > 0 ? dBox[0] * webcamHeight : 0;
        const startX = dBox[1] > 0 ? dBox[1] * webcamWidth : 0;
        const height = (dBox[2] - dBox[0]) / webcamHeight;
        const width = (dBox[3] - dBox[1]) / webcamWidth;
        console.log(height, width);
        ctx.strokeRect(0, 0, width, height);
        // Draw the label background.
        ctx.globalCompositeOperation = "source-over";
        ctx.fillStyle = "transparent";
        const textHeight = 8;
        const textPad = 4;
        const label = `${detectedClass} ${Math.round(detectedScore * 100)}%`;
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(startX, startY, textWidth + textPad, textHeight + textPad);
        // Draw the text last to ensure it's on top.
        ctx.fillStyle = "#00FF00";
        ctx.fillText(label, startX, startY);
      });
    }
    // Loop forever
    requestAnimationFrame(() => {
      this.detect(model, webcamRef, canvasRef);
    });
  }

  public async run(webCamRef: any, canvasRef: any) {
    try {
      const model = await this.loadModel();
      this.detect(model, webCamRef, canvasRef);
    } catch (e) {
      console.error(e);
    }
  }
}

const CLASSES = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "unused",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "unused",
  "backpack",
  "umbrella",
  "unused",
  "unused",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "unused",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "unused",
  "dining table",
  "unused",
  "unused",
  "toilet",
  "unused",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "unused",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
];
