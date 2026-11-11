import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

let poseLandmarker = null;
let lastDetectionTime = -Infinity;
const lastResultsByVideo = new WeakMap();
const MIN_DETECTION_INTERVAL_MS = 100; // Keep CPU inference near 10 FPS total.

export async function initializeMotionTracking() {
  // Reuse one small model for webcam and uploaded-video modes to save memory.
  if (poseLandmarker) return true;
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "CPU"
    },
    runningMode: "VIDEO",
    numPoses: 1,
    outputSegmentationMasks: false,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
  return true;
}

function detectPose(video, timestamp = performance.now()) {
  if (!poseLandmarker || !video || video.readyState < 2) return null;
  const cached = lastResultsByVideo.get(video) || null;
  // A single shared model keeps RAM use down. Cache each input separately so
  // webcam landmarks are never accidentally drawn over the uploaded video.
  if (timestamp - lastDetectionTime < MIN_DETECTION_INTERVAL_MS) return cached;
  lastDetectionTime = timestamp;
  try {
    const result = poseLandmarker.detectForVideo(video, timestamp);
    lastResultsByVideo.set(video, result);
    return result;
  } catch (error) {
    console.warn("Pose detection skipped:", error);
    return cached;
  }
}

export function trackBody(video, timestamp = performance.now()) {
  return detectPose(video, timestamp);
}

export function trackMotionVideo(video, timestamp = performance.now()) {
  return detectPose(video, timestamp);
}

export function drawBodyPose(ctx, landmarks, width, height) {
  ctx.clearRect(0, 0, width, height);
  if (!landmarks || !landmarks.length) return;
  const p = landmarks[0];
  const links = [
    [11,12],[11,13],[13,15],[12,14],[14,16],
    [11,23],[12,24],[23,24],[23,25],[25,27],[27,29],[29,31],
    [24,26],[26,28],[28,30],[30,32]
  ];
  ctx.lineWidth = Math.max(2, width / 320);
  ctx.strokeStyle = "#39e6b0";
  ctx.fillStyle = "#ffe36e";
  for (const [a,b] of links) {
    if (!p[a] || !p[b] || (p[a].visibility ?? 1) < 0.35 || (p[b].visibility ?? 1) < 0.35) continue;
    ctx.beginPath();
    ctx.moveTo(p[a].x * width, p[a].y * height);
    ctx.lineTo(p[b].x * width, p[b].y * height);
    ctx.stroke();
  }
  for (const i of [11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32]) {
    if (!p[i] || (p[i].visibility ?? 1) < 0.35) continue;
    ctx.beginPath();
    ctx.arc(p[i].x * width, p[i].y * height, Math.max(3,width/180), 0, Math.PI*2);
    ctx.fill();
  }
}
