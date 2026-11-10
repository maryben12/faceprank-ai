
import {
  FaceLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

let faceLandmarker = null;

export async function initializeFaceTracking() {
  const status = document.getElementById("status");

  try {
    status.textContent = "Loading AI face tracker...";

    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    faceLandmarker = await FaceLandmarker.createFromOptions(
      vision,
      {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          delegate: "CPU"
        },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFaceBlendshapes: true
      }
    );

    status.textContent = "AI face tracker ready!";
    return true;

  } catch (error) {
    console.error(error);
    status.textContent = "Could not load AI tracker. Check internet.";
    return false;
  }
}

export function trackFace(video) {
  if (!faceLandmarker || video.readyState < 2) {
    return null;
  }

  return faceLandmarker.detectForVideo(
    video,
    performance.now()
  );
}
