import {
  FaceLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

let videoLandmarker = null;
let imageLandmarker = null;

async function createLandmarker(vision, runningMode) {
  return FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "CPU"
    },
    runningMode,
    numFaces: 1,
    outputFaceBlendshapes: true
  });
}

export async function initializeFaceTracking() {
  const status = document.getElementById("status");
  try {
    status.textContent = "Loading AI face tracker...";
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    videoLandmarker = await createLandmarker(vision, "VIDEO");
    imageLandmarker = await createLandmarker(vision, "IMAGE");
    status.textContent = "AI face tracker ready!";
    return true;
  } catch (error) {
    console.error(error);
    status.textContent = "Could not load AI tracker. Check your internet connection.";
    return false;
  }
}

export function trackFace(video) {
  if (!videoLandmarker || video.readyState < 2) return null;
  return videoLandmarker.detectForVideo(video, performance.now());
}

export function detectPhoto(image) {
  if (!imageLandmarker || !image || !image.complete) return null;
  return imageLandmarker.detect(image);
}
