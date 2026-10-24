import { useEffect, useRef, useState } from "react";
import type { Niivue } from "@niivue/niivue";
import { Brain, AlertCircle, Boxes, Layers, Wand2 } from "lucide-react";
import {
  getCaseViewerVolume,
  listCaseViewerCases,
  messageOf,
  segmentCaseViewerCase,
  type CaseViewerCase,
  type SegmentationResult,
} from "../lib/api";

type MaskSource = "expert" | "model";

const REGION_META: { key: "ncr" | "ed" | "et"; label: string; color: string }[] = [
  { key: "et", label: "ET · enhancing", color: "#ef4444" },
  { key: "ed", label: "ED · edema", color: "#10b981" },
  { key: "ncr", label: "NCR · necrosis", color: "#f59e0b" },
];

// BraTS labels 1=NCR, 2=ED, 4=ET: discrete label colormap for NiiVue
const SEG_CMAP = {
  R: [0, 230, 60, 230],
  G: [0, 180, 230, 60],
  B: [0, 60, 60, 60],
  A: [0, 255, 255, 255],
  I: [0, 1, 2, 4],
  labels: ["bg", "NCR", "ED", "ET"],
};

/** Load the T1-CE base plus the chosen mask. Volumes are fetched with the Bearer token and handed to NiiVue as blob URLs. */
async function loadCaseVolumes(nv: Niivue, caseId: string, mask: MaskSource): Promise<void> {
  const [baseBlob, maskBlob] = await Promise.all([
    getCaseViewerVolume(caseId, "t1ce"),
    getCaseViewerVolume(caseId, mask === "expert" ? "seg" : "pred"),
  ]);
  const urls = [URL.createObjectURL(baseBlob), URL.createObjectURL(maskBlob)];
  try {
    await nv.loadVolumes([
      { url: urls[0], name: "t1ce.nii.gz" },
      // The model mask is served uncompressed, the expert mask gzipped; NiiVue keys on the name.
      { url: urls[1], name: mask === "expert" ? "seg.nii.gz" : "seg.nii" },
    ]);
  } finally {
    urls.forEach((url) => URL.revokeObjectURL(url));
  }
  const seg = nv.volumes[1];
  if (seg) {
    seg.setColormapLabel(SEG_CMAP);
    seg.opacity = 0.9;
  }
}

function applyView(nv: Niivue, view3d: boolean): void {
  nv.setSliceType(view3d ? nv.sliceTypeRender : nv.sliceTypeMultiplanar);
  if (view3d) nv.setRenderAzimuthElevation(45, 20);
  nv.drawScene();
}

const pct = (part: number, total: number) =>
  total > 0 ? ((part / total) * 100).toFixed(1) : "0.0";

export function CaseViewer() {
  const [cases, setCases] = useState<CaseViewerCase[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<CaseViewerCase | null>(null);
  const [loadState, setLoadState] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [view3d, setView3d] = useState(true);
  const [segState, setSegState] = useState<"idle" | "running" | "done" | "error">("idle");
  const [maskSource, setMaskSource] = useState<MaskSource>("expert");
  const [maskBusy, setMaskBusy] = useState(false);
  const [predMetrics, setPredMetrics] = useState<SegmentationResult | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const nvRef = useRef<Niivue | null>(null);

  useEffect(() => {
    listCaseViewerCases()
      .then(setCases)
      .catch((err) => setError(messageOf(err, "Failed to load cases")))
      .finally(() => setLoading(false));
  }, []);

  // Runs after the canvas is mounted: create a NiiVue instance per selected case
  // (the canvas is a new element each time) and release its WebGL context on leave.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!selected || !canvas) return;
    let cancelled = false;
    let nv: Niivue | null = null;

    setLoadState("loading");
    setView3d(true);
    setMaskSource("expert");
    setPredMetrics(null);
    setSegState("idle");

    (async () => {
      try {
        const { Niivue: NiivueCtor } = await import("@niivue/niivue");
        // Dynamic on purpose: keeps the ~2 MB WebGL library out of the main bundle until a case is opened.
        if (cancelled) return;
        nv = new NiivueCtor({ dragAndDropEnabled: false });
        await nv.attachToCanvas(canvas);
        if (cancelled) return;
        await loadCaseVolumes(nv, selected.case_id, "expert");
        if (cancelled) return;
        nvRef.current = nv;
        applyView(nv, true);
        setLoadState("done");
      } catch (err) {
        if (cancelled) return;
        setError(messageOf(err, "Failed to load volume"));
        setLoadState("error");
      }
    })();

    return () => {
      cancelled = true;
      nv?.cleanup();
      nvRef.current = null;
    };
  }, [selected]);

  const toggleView = (to3d: boolean) => {
    const nv = nvRef.current;
    if (!nv) return;
    setView3d(to3d);
    applyView(nv, to3d);
  };

  /** Swap the mask volume; ignores the result if the case was closed meanwhile. */
  const showMask = async (nv: Niivue, caseId: string, source: MaskSource) => {
    await loadCaseVolumes(nv, caseId, source);
    if (nvRef.current !== nv) return false;
    setMaskSource(source);
    applyView(nv, view3d);
    return true;
  };

  const runSegmentation = async () => {
    const nv = nvRef.current;
    if (!nv || !selected || segState === "running" || maskBusy) return;
    setSegState("running");
    setError("");
    try {
      const metrics = await segmentCaseViewerCase(selected.case_id);
      if (nvRef.current !== nv) return;
      setPredMetrics(metrics);
      if (await showMask(nv, selected.case_id, "model")) setSegState("done");
    } catch (err) {
      if (nvRef.current !== nv) return;
      setError(messageOf(err, "Segmentation failed"));
      setSegState("error");
    }
  };

  const switchMask = async (source: MaskSource) => {
    const nv = nvRef.current;
    if (!nv || !selected || source === maskSource || maskBusy || segState === "running") return;
    if (source === "model" && !predMetrics) {
      await runSegmentation();
      return;
    }
    setMaskBusy(true);
    setError("");
    try {
      await showMask(nv, selected.case_id, source);
    } catch (err) {
      if (nvRef.current === nv) setError(messageOf(err, "Failed to switch mask"));
    } finally {
      setMaskBusy(false);
    }
  };

  const volumes =
    selected && maskSource === "model" && predMetrics
      ? predMetrics.volumes_cm3
      : selected?.volumes_cm3;

  return (
    <div className="page-container" style={{ maxWidth: "1400px" }}>
      <h1 className="page-title">
        <Brain size={24} style={{ color: "var(--accent)" }} /> Case Viewer — 3D Tumor Analysis
      </h1>
      <p className="text-muted" style={{ marginBottom: "1.5rem" }}>
        BraTS 2021 sample volumes · multi-region masks (ET · ED · NCR) · real volume in cm³ · drag to rotate in 3D
      </p>

      {error && (
        <div className="auth-error">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      {loading && (
        <div className="loading-screen" style={{ minHeight: "20rem" }}>
          <div className="spinner" />
        </div>
      )}

      {!loading && !selected && (
        <div className="stats" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
          {cases.map((c) => (
            <button
              key={c.case_id}
              className="card action-card"
              style={{ textAlign: "left", padding: "1rem", background: "#fff" }}
              onClick={() => { setError(""); setSelected(c); }}
            >
              <p className="eyebrow" style={{ margin: 0 }}>CASE</p>
              <h3 style={{ margin: "0.25rem 0" }}>{c.case_id}</h3>
              <p className="text-muted" style={{ margin: 0, fontSize: "0.85rem" }}>
                <strong style={{ color: "var(--accent)" }}>{c.volumes_cm3.total.toFixed(1)} cm³</strong>{" "}
                tumor · {c.slices} slices
              </p>
            </button>
          ))}
        </div>
      )}

      {selected && volumes && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 300px", gap: "1.25rem", alignItems: "start" }}>
          <div className="card" style={{ padding: "1rem", background: "#fff" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
              <button className="btn btn-tiny" onClick={() => { setSelected(null); setLoadState("idle"); }}>
                ← All cases
              </button>
              <div style={{ display: "flex", gap: "0.4rem" }}>
                <button className={`btn btn-tiny ${view3d ? "active" : ""}`} onClick={() => toggleView(true)}>3D</button>
                <button className={`btn btn-tiny ${!view3d ? "active" : ""}`} onClick={() => toggleView(false)}>Slices</button>
              </div>
              <strong>{selected.case_id}</strong>
            </div>

            <div style={{ position: "relative", background: "#0f1115", borderRadius: "6px", overflow: "hidden" }}>
              <canvas ref={canvasRef} style={{ width: "100%", height: "calc(100vh - 15rem)", minHeight: "28rem", display: "block" }} />
              {loadState === "loading" && (
                <div className="loading-screen" style={{ position: "absolute", inset: 0, background: "rgba(15,17,21,0.7)" }}>
                  <div className="spinner" />
                  <p style={{ color: "#94a3b8", fontSize: "0.85rem" }}>Loading volume…</p>
                </div>
              )}
              {loadState === "error" && (
                <div className="loading-screen" style={{ position: "absolute", inset: 0, background: "rgba(15,17,21,0.7)" }}>
                  <p style={{ color: "#f87171", fontSize: "0.85rem" }}>Could not load volume</p>
                </div>
              )}
            </div>
            <p className="text-muted" style={{ fontSize: "0.75rem", marginTop: "0.5rem", display: "flex", gap: "0.4rem", alignItems: "center" }}>
              <Boxes size={14} /> left drag = rotate · scroll = zoom · right drag = pan
            </p>
          </div>

          <div className="card" style={{ padding: "1rem", background: "#fff" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <p className="eyebrow" style={{ marginTop: 0 }}>TUMOR COMPOSITION</p>
              <div style={{ display: "flex", gap: "0.3rem" }}>
                <button
                  className={`btn btn-tiny ${maskSource === "expert" ? "active" : ""}`}
                  onClick={() => switchMask("expert")}
                  disabled={maskBusy || segState === "running"}
                >
                  Expert
                </button>
                <button
                  className={`btn btn-tiny ${maskSource === "model" ? "active" : ""}`}
                  onClick={() => switchMask("model")}
                  disabled={maskBusy || segState === "running"}
                >
                  Model
                </button>
              </div>
            </div>
            {REGION_META.map(({ key, label, color }) => (
              <div key={key} style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
                <span style={{ width: 12, height: 12, borderRadius: 3, background: color }} />
                <span style={{ flex: 1 }}>{label}</span>
                <strong>{volumes[key].toFixed(1)} cm³</strong>
                <span className="text-muted" style={{ width: "3rem", textAlign: "right" }}>
                  {pct(volumes[key], volumes.total)}%
                </span>
              </div>
            ))}
            <div style={{ borderTop: "1px solid rgba(45,49,66,0.12)", paddingTop: "0.75rem", display: "flex", justifyContent: "space-between" }}>
              <span>Total</span>
              <strong style={{ color: "var(--accent)" }}>{volumes.total.toFixed(1)} cm³</strong>
            </div>

            {maskSource === "expert" ? (
              <button className="btn btn-primary btn-block" style={{ marginTop: "1rem" }} onClick={runSegmentation} disabled={segState === "running" || maskBusy}>
                <Wand2 size={16} /> {segState === "running" ? "Segmenting… ≈2 min" : "3D segmentation · SwinUNETR"}
              </button>
            ) : (
              <div className="text-muted" style={{ marginTop: "0.75rem", fontSize: "0.8rem" }}>
                Mask from the SwinUNETR model{predMetrics ? ` · ${predMetrics.seconds}s GPU` : ""}
              </div>
            )}
            <div className="text-muted" style={{ marginTop: "0.75rem", fontSize: "0.8rem", display: "flex", gap: "0.4rem", alignItems: "center" }}>
              <Layers size={14} /> {maskSource === "expert" ? "expert segmentation mask" : "predicted mask (inference ~2 min, cached)"}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
