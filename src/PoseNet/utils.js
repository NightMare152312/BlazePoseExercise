import * as posenet from '@tensorflow-models/posenet';
import * as poseDetection from '@tensorflow-models/pose-detection';

const model = poseDetection.SupportedModels.BlazePose;

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

export function isMobile() {
  return isAndroid() || isiOS();
}

export function drawKeypoints(keypoints, minConfidence, skeletonColor, ctx, scale = 1, skipIndex) {
  keypoints.forEach((keypoint, index) => {
    if(skipIndex.includes(index)){
      return;
    }
    if (keypoint.score >= minConfidence) {
      const { y, x } = keypoint
      ctx.beginPath()
      ctx.arc(x * scale, y * scale, 3, 0, 2 * Math.PI)
      ctx.fillStyle = skeletonColor
      ctx.fill()
    }
  })
}

function toTuple({ y, x }) {
  return [y, x]
}

function drawSegment([ay, ax], [by, bx], color, lineWidth, scale, ctx) {
  ctx.beginPath()
  ctx.moveTo(ax * scale, ay * scale)
  ctx.lineTo(bx * scale, by * scale)
  ctx.lineWidth = lineWidth
  ctx.strokeStyle = color
  ctx.stroke()
}

export function drawSkeleton(keypoints, minConfidence, color, lineWidth, ctx, scale = 1, skipIndex) {
  // const adjacentKeyPoints = posenet.getAdjacentKeyPoints(keypoints, minConfidence)
  const adjacentKeyPoints = poseDetection.util.getAdjacentPairs(model)

  adjacentKeyPoints.forEach(([i,j]) => {
    if (skipIndex.includes(i) || skipIndex.includes(j)) {
      return;
    }

    const kp1 = keypoints[i];
    const kp2 = keypoints[j];
    if(kp1.score >= minConfidence && kp2.score >= minConfidence){
      drawSegment(
        toTuple(kp1),
        toTuple(kp2),
        color, lineWidth, scale, ctx
      )      
    } 
  })
}
