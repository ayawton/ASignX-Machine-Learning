/**
 * ASignX Client-Side Web SDK for ASignLearn & ASignLive
 *
 * Runs 100% in the user's browser using @mediapipe/tasks-vision (WebGL) and
 * onnxruntime-web (WebAssembly/WebGPU) with zero video server round-trips.
 *
 * Features:
 *  - Exact TypeScript port of ASignX wrist-relative + max-distance normalization
 *  - 91-dimensional Kinematic Feature Enricher (joint cosines, tip distances, palm normal)
 *  - MotionGate velocity router separating static holds from dynamic trajectories
 *  - Hysteresis temporal stabilizer (stableFrames = 5, releaseFrames = 10) for ASignLive
 *  - Target sign probability gauge + per-finger error diagnostics against class_centroids
 *    for ASignLearn interactive finger coaching
 */

export interface Point3D {
  x: number;
  y: number;
  z: number;
}

export interface ASignXModelMetadata {
  model_type: string;
  labels: string[];
  input_dim: number;
  hidden_dim: number;
  use_kinematic_features: boolean;
  use_cosine_head: boolean;
  class_centroids?: Record<string, number[]>;
  static_hand_preprocessing?: "none" | "left_x_reflection_v1";
}

export interface FingerCoachingFeedback {
  finger: "Thumb" | "Index" | "Middle" | "Ring" | "Pinky";
  meanError: number;
  isAligned: boolean;
  hint: string;
}

export interface ASignLearnDiagnostic {
  targetLabel: string;
  targetProbability: number;
  predictedLabel: string | null;
  holdProgress: number; // [0.0, 1.0]
  passed: boolean;
  fingerFeedback: FingerCoachingFeedback[];
}

export interface ASignLiveRecognition {
  hand: "Left" | "Right";
  predictedLabel: string | null;
  confidence: number;
  stableLabel: string | null;
  emittedLabel: string | null;
  isMoving: boolean;
  velocity: number;
}

const FINGER_INDICES: Record<FingerCoachingFeedback["finger"], number[]> = {
  Thumb: [1, 2, 3, 4],
  Index: [5, 6, 7, 8],
  Middle: [9, 10, 11, 12],
  Ring: [13, 14, 15, 16],
  Pinky: [17, 18, 19, 20],
};

const JOINT_TRIPLETS: [number, number, number][] = [
  [0, 1, 2], [1, 2, 3], [2, 3, 4],
  [0, 5, 6], [5, 6, 7], [6, 7, 8],
  [0, 9, 10], [9, 10, 11], [10, 11, 12],
  [0, 13, 14], [13, 14, 15], [14, 15, 16],
  [0, 17, 18], [17, 18, 19], [18, 19, 20],
];

const FINGERTIP_PAIRS: [number, number][] = [
  [4, 8], [4, 12], [4, 16], [4, 20],
  [8, 12], [8, 16], [8, 20],
  [12, 16], [12, 20],
  [16, 20],
];

/**
 * Normalize 21 MediaPipe 3D hand landmarks into a 63-float array:
 * Translate wrist to zero, scale by maximum radius, then optionally reflect Left x.
 * Pass the exported metadata mode; missing metadata means legacy normalization.
 * Use detector coordinates and handedness from the same image; CSS mirroring is display-only.
 */
export function normalizeHandLandmarks(
  landmarks: Point3D[],
  handedness?: "Left" | "Right",
  mode: "none" | "left_x_reflection_v1" = "none"
): Float32Array {
  if (landmarks.length !== 21) {
    throw new Error(`Expected 21 landmarks, received ${landmarks.length}`);
  }
  const coords = new Float32Array(63);
  if (mode !== "none" && mode !== "left_x_reflection_v1") {
    throw new Error(`Unsupported static hand preprocessing: ${mode}`);
  }
  if (mode !== "none" && handedness !== "Left" && handedness !== "Right") {
    throw new Error("Static hand canonicalization requires Left or Right handedness");
  }

  for (let i = 0; i < 21; i++) {
    coords[i * 3] = landmarks[i].x;
    coords[i * 3 + 1] = landmarks[i].y;
    coords[i * 3 + 2] = landmarks[i].z;
    if (![coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]].every(Number.isFinite)) {
      throw new Error("Expected finite landmark coordinates");
    }
  }

  const wx = coords[0];
  const wy = coords[1];
  const wz = coords[2];

  let maxDist = 0.0;
  for (let i = 0; i < 21; i++) {
    const dx = coords[i * 3] - wx;
    const dy = coords[i * 3 + 1] - wy;
    const dz = coords[i * 3 + 2] - wz;
    coords[i * 3] = dx;
    coords[i * 3 + 1] = dy;
    coords[i * 3 + 2] = dz;
    const dist = Math.hypot(dx, dy, dz);
    if (dist > maxDist) {
      maxDist = dist;
    }
  }

  if (maxDist <= 1e-8) {
    throw new Error("Degenerate hand landmarks: max distance from wrist is zero");
  }

  for (let i = 0; i < 63; i++) {
    coords[i] /= maxDist;
    if (mode === "left_x_reflection_v1" && handedness === "Left" && i % 3 === 0) {
      coords[i] *= -1;
    }
    if (coords[i] === 0) coords[i] = 0;
  }
  return coords;
}

/** Prepare the 63D graph input (and coaching input) using this checkpoint's contract. */
export function prepareStaticHandLandmarks(
  landmarks: Point3D[],
  handedness: "Left" | "Right" | undefined,
  metadata: ASignXModelMetadata
): Float32Array {
  return normalizeHandLandmarks(landmarks, handedness, metadata.static_hand_preprocessing ?? "none");
}

/**
 * Enrich 63D normalized coordinates into a 91D biomechanical feature vector:
 * [63 coords, 15 joint flexion angle cosines, 10 fingertip distances, 3 palm normal components]
 */
export function computeKinematicFeatures91D(normalized63D: Float32Array): Float32Array {
  const out = new Float32Array(91);
  out.set(normalized63D, 0);
  let offset = 63;

  // 15 joint flexion angle cosines
  for (const [a, b, c] of JOINT_TRIPLETS) {
    const v1x = normalized63D[a * 3] - normalized63D[b * 3];
    const v1y = normalized63D[a * 3 + 1] - normalized63D[b * 3 + 1];
    const v1z = normalized63D[a * 3 + 2] - normalized63D[b * 3 + 2];

    const v2x = normalized63D[c * 3] - normalized63D[b * 3];
    const v2y = normalized63D[c * 3 + 1] - normalized63D[b * 3 + 1];
    const v2z = normalized63D[c * 3 + 2] - normalized63D[b * 3 + 2];

    const denom = Math.max(Math.hypot(v1x, v1y, v1z) * Math.hypot(v2x, v2y, v2z), 1e-6);
    const cosVal = Math.max(-1.0, Math.min(1.0, (v1x * v2x + v1y * v2y + v1z * v2z) / denom));
    out[offset++] = cosVal;
  }

  // 10 pairwise fingertip Euclidean distances
  for (const [i, j] of FINGERTIP_PAIRS) {
    const dx = normalized63D[i * 3] - normalized63D[j * 3];
    const dy = normalized63D[i * 3 + 1] - normalized63D[j * 3 + 1];
    const dz = normalized63D[i * 3 + 2] - normalized63D[j * 3 + 2];
    out[offset++] = Math.hypot(dx, dy, dz);
  }

  // 3D unit palm normal vector: cross(index_mcp - wrist, pinky_mcp - wrist)
  const ux = normalized63D[5 * 3] - normalized63D[0];
  const uy = normalized63D[5 * 3 + 1] - normalized63D[1];
  const uz = normalized63D[5 * 3 + 2] - normalized63D[2];

  const vx = normalized63D[17 * 3] - normalized63D[0];
  const vy = normalized63D[17 * 3 + 1] - normalized63D[1];
  const vz = normalized63D[17 * 3 + 2] - normalized63D[2];

  const nx = uy * vz - uz * vy;
  const ny = uz * vx - ux * vz;
  const nz = ux * vy - uy * vx;
  const nNorm = Math.max(Math.hypot(nx, ny, nz), 1e-6);

  out[offset++] = nx / nNorm;
  out[offset++] = ny / nNorm;
  out[offset++] = nz / nNorm;

  return out;
}

export function softmax(logits: Float32Array | number[]): Float32Array {
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > maxLogit) maxLogit = logits[i];
  }
  const probs = new Float32Array(logits.length);
  let sum = 0.0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - maxLogit);
    probs[i] = e;
    sum += e;
  }
  for (let i = 0; i < logits.length; i++) {
    probs[i] /= sum;
  }
  return probs;
}

/**
 * ASignLearn Coach: Computes target sign probability, hold progress, and per-finger
 * green/red alignment diagnostics against the exported class centroid.
 */
export class ASignLearnCoach {
  private holdCounter = 0;
  private readonly metadata: ASignXModelMetadata;
  private readonly requiredHoldFrames: number;
  private readonly passConfidence: number;
  private readonly fingerErrorTolerance: number;

  constructor(
    metadata: ASignXModelMetadata,
    requiredHoldFrames: number = 15,
    passConfidence: number = 0.75,
    fingerErrorTolerance: number = 0.14
  ) {
    this.metadata = metadata; this.requiredHoldFrames = requiredHoldFrames;
    this.passConfidence = passConfidence; this.fingerErrorTolerance = fingerErrorTolerance;
  }

  public evaluateAttempt(
    normalized63D: Float32Array,
    logits: Float32Array,
    targetLabel: string
  ): ASignLearnDiagnostic {
    const probs = softmax(logits);
    const targetIdx = this.metadata.labels.indexOf(targetLabel.toUpperCase());
    const targetProbability = targetIdx >= 0 ? probs[targetIdx] : 0.0;

    let bestIdx = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[bestIdx]) bestIdx = i;
    }
    const predictedLabel =
      probs[bestIdx] >= 0.5 ? this.metadata.labels[bestIdx] : null;

    if (targetProbability >= this.passConfidence) {
      this.holdCounter = Math.min(this.requiredHoldFrames, this.holdCounter + 1);
    } else {
      this.holdCounter = Math.max(0, this.holdCounter - 1);
    }

    const holdProgress = this.holdCounter / this.requiredHoldFrames;
    const passed = this.holdCounter >= this.requiredHoldFrames;

    const fingerFeedback: FingerCoachingFeedback[] = [];
    const centroid = this.metadata.class_centroids?.[targetLabel.toUpperCase()];

    for (const [finger, jointIndices] of Object.entries(FINGER_INDICES) as [
      FingerCoachingFeedback["finger"],
      number[]
    ][]) {
      let meanError = 0.0;
      if (centroid && centroid.length === 63) {
        let totalDist = 0.0;
        for (const idx of jointIndices) {
          const dx = normalized63D[idx * 3] - centroid[idx * 3];
          const dy = normalized63D[idx * 3 + 1] - centroid[idx * 3 + 1];
          const dz = normalized63D[idx * 3 + 2] - centroid[idx * 3 + 2];
          totalDist += Math.hypot(dx, dy, dz);
        }
        meanError = totalDist / jointIndices.length;
      }

      const isAligned = meanError <= this.fingerErrorTolerance;
      fingerFeedback.push({
        finger,
        meanError,
        isAligned,
        hint: isAligned
          ? `${finger} position looks great!`
          : `Adjust your ${finger.toLowerCase()} finger to match the '${targetLabel}' reference pose.`,
      });
    }

    return {
      targetLabel: targetLabel.toUpperCase(),
      targetProbability,
      predictedLabel,
      holdProgress,
      passed,
      fingerFeedback,
    };
  }

  public reset(): void {
    this.holdCounter = 0;
  }
}

/**
 * ASignLive Temporal Stabilizer + MotionGate:
 * Prevents duplicate emissions and suppresses static false positives during fast motion.
 */
export class ASignLiveStabilizer {
  private candidateLabel: string | null = null;
  private candidateStreak = 0;
  private stableLabel: string | null = null;
  private lastEmittedLabel: string | null = null;
  private missingStreak = 0;
  private prevWrist: Point3D | null = null;
  private motionHistory: { points: Point3D[]; timestamp: number }[] = [];
  private readonly metadata: ASignXModelMetadata;
  private readonly confidenceThreshold: number;
  private readonly stableFrames: number;
  private readonly releaseFrames: number;
  private readonly motionThreshold: number;

  constructor(
    metadata: ASignXModelMetadata,
    confidenceThreshold: number = 0.8,
    stableFrames: number = 5,
    releaseFrames: number = 10,
    motionThreshold: number = 0.045
  ) {
    this.metadata = metadata; this.confidenceThreshold = confidenceThreshold;
    this.stableFrames = stableFrames; this.releaseFrames = releaseFrames; this.motionThreshold = motionThreshold;
  }

  public update(
    hand: "Left" | "Right",
    rawLandmarks: Point3D[],
    logits: Float32Array,
    timestamp: number = performance.now() / 1000
  ): ASignLiveRecognition {
    // Match Python: wrist + fingertips over a time-bounded history, at reference 30 FPS.
    const wrist = rawLandmarks[0];
    let velocity = 0.0;
    const previous = this.motionHistory[this.motionHistory.length - 1];
    if (!Number.isFinite(timestamp) || (previous && timestamp <= previous.timestamp)) {
      throw new Error("Motion timestamps must increase and be finite");
    }
    if (previous && timestamp - previous.timestamp > 0.5) this.motionHistory = [];
    this.motionHistory.push({ points: [0, 4, 8, 12, 16, 20].map(i => ({ ...rawLandmarks[i] })), timestamp });
    if (this.motionHistory.length > 128) this.motionHistory.shift();
    while (this.motionHistory.length > 2 && timestamp - this.motionHistory[0].timestamp > 7 / 30) this.motionHistory.shift();
    for (let i = 1; i < this.motionHistory.length; i++) {
      const a = this.motionHistory[i-1], b = this.motionHistory[i];
      velocity += b.points.reduce((total, p, j) => total + Math.hypot(p.x-a.points[j].x, p.y-a.points[j].y, p.z-a.points[j].z), 0) / 6;
    }
    if (this.motionHistory.length > 1) velocity /= 30 * (timestamp - this.motionHistory[0].timestamp);
    this.prevWrist = { ...wrist };
    const isMoving = velocity >= this.motionThreshold;

    const probs = softmax(logits);
    let bestIdx = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[bestIdx]) bestIdx = i;
    }
    const confidence = probs[bestIdx];
    const predictedLabel =
      !isMoving && confidence >= this.confidenceThreshold
        ? this.metadata.labels[bestIdx]
        : null;

    this.missingStreak = predictedLabel === null ? this.missingStreak + 1 : 0;
    let emittedLabel: string | null = null;

    if (predictedLabel === null) {
      this.candidateLabel = null;
      this.candidateStreak = 0;
      this.stableLabel = null;
      if (this.missingStreak >= this.releaseFrames) this.lastEmittedLabel = null;
    } else if (predictedLabel === this.candidateLabel) {
      this.candidateStreak += 1;
    } else {
      this.candidateLabel = predictedLabel;
      this.candidateStreak = 1;
    }

    if (this.candidateLabel !== null && this.candidateStreak >= this.stableFrames) {
      this.stableLabel = this.candidateLabel;
      if (this.stableLabel !== this.lastEmittedLabel) {
        emittedLabel = this.stableLabel;
        this.lastEmittedLabel = this.stableLabel;
      }
    }

    return {
      hand,
      predictedLabel,
      confidence,
      stableLabel: this.stableLabel,
      emittedLabel,
      isMoving,
      velocity,
    };
  }

  public markHandMissing(): void {
    this.missingStreak += 1;
    this.candidateLabel = null;
    this.candidateStreak = 0;
    this.prevWrist = null;
    this.motionHistory = [];
    if (this.missingStreak >= this.releaseFrames) {
      this.stableLabel = null;
      this.lastEmittedLabel = null;
    }
  }
}

