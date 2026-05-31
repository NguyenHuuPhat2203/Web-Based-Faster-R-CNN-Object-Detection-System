import { useState, useRef, useEffect, useCallback } from "react";
import { Camera, CameraOff, AlertCircle, RotateCcw } from "lucide-react";
import { predict, type PredictionResult, type ModelType } from "../lib/api";

/* ---------- Types ---------- */

interface WebcamCaptureProps {
  model: ModelType;
  threshold: number;
  onResult: (result: PredictionResult) => void;
  onError: (error: string) => void;
}

type WebcamState = "init" | "requesting" | "active" | "denied" | "error";

/* ---------- Component ---------- */

export function WebcamCapture({ model, threshold, onResult, onError }: WebcamCaptureProps) {
  const [state, setState] = useState<WebcamState>("init");
  const [capturing, setCapturing] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasOverlayRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number>(0);

  /* ── Start camera ─────────────────────────────────────────────── */

  const startCamera = useCallback(async () => {
    setState("requesting");
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      setStream(s);
      setState("active");
    } catch (err: any) {
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        setState("denied");
      } else {
        setState("error");
      }
    }
  }, []);

  /* ── Stop camera ──────────────────────────────────────────────── */

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      setStream(null);
    }
    cancelAnimationFrame(animFrameRef.current);
    setState("init");
  }, [stream]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (stream) stream.getTracks().forEach((t) => t.stop());
      cancelAnimationFrame(animFrameRef.current);
    };
  }, [stream]);

  /* ── Capture frame and predict ────────────────────────────────── */

  const captureAndPredict = useCallback(async () => {
    const video = videoRef.current;
    if (!video || capturing) return;

    setCapturing(true);
    try {
      // Draw the current video frame to a temp canvas → blob
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = video.videoWidth;
      tempCanvas.height = video.videoHeight;
      const tempCtx = tempCanvas.getContext("2d");
      if (!tempCtx) return;
      tempCtx.drawImage(video, 0, 0);

      const blob = await new Promise<Blob | null>((resolve) =>
        tempCanvas.toBlob(resolve, "image/jpeg", 0.85),
      );
      if (!blob) return;

      const file = new File([blob], `webcam-${Date.now()}.jpg`, { type: "image/jpeg" });
      const result = await predict(file, threshold, model);

      // Draw overlay
      const overlay = canvasOverlayRef.current;
      if (overlay) {
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
        const ctx = overlay.getContext("2d");
        if (ctx) {
          ctx.clearRect(0, 0, overlay.width, overlay.height);
          ctx.lineWidth = 3;
          ctx.font = "bold 16px Inter, system-ui, sans-serif";

          result.boxes.forEach((box, i) => {
            const [xmin, ymin, xmax, ymax] = box;
            const score = result.scores[i];
            const label = result.label_names?.[i] || `Class ${result.labels[i]}`;

            ctx.strokeStyle = "#10b981";
            ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

            const labelText = `${label}: ${(score * 100).toFixed(1)}%`;
            const textWidth = ctx.measureText(labelText).width;

            ctx.fillStyle = "#10b981";
            ctx.fillRect(xmin, ymin - 28, textWidth + 10, 28);

            ctx.fillStyle = "white";
            ctx.fillText(labelText, xmin + 5, ymin - 7);
          });
        }
      }

      onResult(result);
    } catch (err: any) {
      onError(err.message || "Detection failed");
    } finally {
      setCapturing(false);
    }
  }, [capturing, threshold, model, onResult, onError]);

  /* ── Attach stream to video element ────────────────────────────── */

  useEffect(() => {
    const video = videoRef.current;
    if (video && stream) {
      video.srcObject = stream;
      video.play().catch(() => {});
    }
  }, [stream]);

  /* ── Render ────────────────────────────────────────────────────── */

  if (state === "init") {
    return (
      <div className="webcam-init">
        <button className="btn btn-primary" onClick={startCamera}>
          <Camera size={20} /> Start Webcam
        </button>
      </div>
    );
  }

  if (state === "requesting") {
    return (
      <div className="webcam-status">
        <div className="spinner" />
        <p>Requesting camera access&hellip;</p>
      </div>
    );
  }

  if (state === "denied") {
    return (
      <div className="webcam-status">
        <CameraOff size={32} />
        <p><strong>Camera access denied</strong></p>
        <p className="text-muted">Please allow camera access in your browser settings and reload.</p>
        <button className="btn btn-secondary" onClick={() => setState("init")}
          style={{ width: "auto", marginTop: "0.5rem" }}>
          <RotateCcw size={16} /> Retry
        </button>
      </div>
    );
  }

  if (state === "error") {
    return (
      <div className="webcam-status">
        <AlertCircle size={32} />
        <p><strong>Camera unavailable</strong></p>
        <p className="text-muted">No camera found or a hardware error occurred.</p>
        <button className="btn btn-secondary" onClick={() => setState("init")}
          style={{ width: "auto", marginTop: "0.5rem" }}>
          <RotateCcw size={16} /> Retry
        </button>
      </div>
    );
  }

  /* ── Active ────────────────────────────────────────────────────── */

  return (
    <div className="webcam-active">
      <div className="webcam-viewfinder">
        <video ref={videoRef} autoPlay playsInline muted className="webcam-video" />
        <canvas ref={canvasOverlayRef} className="webcam-overlay" />
      </div>
      <div className="webcam-controls">
        <button
          className="btn btn-primary"
          onClick={captureAndPredict}
          disabled={capturing}
        >
          {capturing ? (
            <><span className="spinner-small" /> Analyzing&hellip;</>
          ) : (
            <><Camera size={18} /> Capture & Detect</>
          )}
        </button>
        <button className="btn btn-secondary" onClick={stopCamera}>
          Stop Camera
        </button>
      </div>
    </div>
  );
}
