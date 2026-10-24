import { useState, useRef, useEffect, useCallback } from "react";
import { Camera, CameraOff, AlertCircle, RotateCcw } from "lucide-react";
import { messageOf, predict, type PredictionResult, type ModelType } from "../lib/api";
import { drawDetections } from "../lib/drawScan";

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
  // Last analysed frame; boxes are in that frame's pixel space.
  const [shot, setShot] = useState<{ result: PredictionResult; width: number; height: number } | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasOverlayRef = useRef<HTMLCanvasElement | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  /* ── Start camera ─────────────────────────────────────────────── */

  const startCamera = useCallback(async () => {
    setState("requesting");
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      if (!mountedRef.current) {
        s.getTracks().forEach((t) => t.stop()); // unmounted while the permission prompt was open
        return;
      }
      setStream(s);
      setState("active");
    } catch (err) {
      if (!mountedRef.current) return;
      const name = err instanceof DOMException ? err.name : "";
      if (name === "NotAllowedError" || name === "PermissionDeniedError") {
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
    setShot(null);
    setState("init");
  }, [stream]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, [stream]);

  /* ── Capture frame and predict ────────────────────────────────── */

  const captureAndPredict = useCallback(async () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth || capturing) return;

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
      const result = await predict(file, threshold, model, { explain: false });

      if (!mountedRef.current) return;
      setShot({ result, width: video.videoWidth, height: video.videoHeight });
      onResult(result);
    } catch (err) {
      if (mountedRef.current) onError(messageOf(err, "Detection failed"));
    } finally {
      if (mountedRef.current) setCapturing(false);
    }
  }, [capturing, threshold, model, onResult, onError]);

  /* ── Overlay: redraw on capture and whenever the confidence slider moves ── */

  useEffect(() => {
    const overlay = canvasOverlayRef.current;
    const ctx = overlay?.getContext("2d");
    if (!overlay || !ctx || !shot) return;
    overlay.width = shot.width; // also clears the previous drawing
    overlay.height = shot.height;
    drawDetections(ctx, { ...shot.result, threshold });
  }, [shot, threshold, state]);

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
