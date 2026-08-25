import React, { useEffect, useRef, useState } from "react";
import type { Niivue } from "@niivue/niivue";
import { Brain, AlertCircle, Boxes, Layers, Wand2 } from "lucide-react";
import {
  listCaseViewerCases,
  getCaseViewerVolume,
  segmentCaseViewerCase,
  type CaseViewerCase,
} from "../lib/api";

// CDN UMD (cùng bản đã chạy ổn ở standalone) — bundle ESM npm hỏng trong
// môi trường browser hạn chế của user (File.stream() cũng thiếu).
function getNiivueCtor(): typeof Niivue {
  const ns = (window as any).niivue;
  if (!ns || !ns.Niivue) throw new Error("NiiVue CDN chưa tải — kiểm tra mạng");
  return ns.Niivue;
}

const REGION_META: { key: "ncr" | "ed" | "et"; label: string; color: string }[] = [
  { key: "et", label: "ET · enhancing", color: "#ef4444" },
  { key: "ed", label: "ED · edema", color: "#10b981" },
  { key: "ncr", label: "NCR · necrosis", color: "#f59e0b" },
];

// BraTS labels 1=NCR, 2=ED, 4=ET — discrete label colormap cho NiiVue
const SEG_CMAP = {
  R: [0, 230, 60, 230],
  G: [0, 180, 230, 60],
  B: [0, 60, 60, 60],
  A: [0, 255, 255, 255],
  I: [0, 1, 2, 4],
  labels: ["bg", "NCR", "ED", "ET"],
};

export function CaseViewer() {
  const [cases, setCases] = useState<CaseViewerCase[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<CaseViewerCase | null>(null);
  const [loadState, setLoadState] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [view3d, setView3d] = useState(true);
  const [segState, setSegState] = useState<"idle" | "running" | "done" | "error">("idle");
  const [maskSource, setMaskSource] = useState<"expert" | "model">("expert");
  const [predMetrics, setPredMetrics] = useState<{ seconds: number; volumes_cm3: Record<string, number> } | null>(null);
  const [diag, setDiag] = useState("");
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const nvRef = useRef<Niivue | null>(null);

  useEffect(() => {
    listCaseViewerCases()
      .then(setCases)
      .catch((err: any) => setError(err.message || "Failed to load cases"))
      .finally(() => setLoading(false));
  }, []);

  // chạy sau khi canvas xuất hiện (post-render) — init NiiVue + load volume
  useEffect(() => {
    if (!selected || !canvasRef.current) return;
    setLoadState("loading");
    if (!nvRef.current) {
      const NiivueCtor = getNiivueCtor();
      const nv = new NiivueCtor({ dragAndDropEnabled: false });
      // colormap + render áp khi mỗi volume load xong (timing proven ở standalone)
      (nv as any).onVolumeLoaded = (vol: any) => {
        if (nv.volumes && nv.volumes.length > 1) {
          const seg = nv.volumes[nv.volumes.length - 1];
          seg.setColormapLabel(SEG_CMAP);
          seg.opacity = 0.9;
        }
        const segVol = nv.volumes && nv.volumes[1];
        setDiag(
          `volumes=${nv.volumes?.length} colormap=${!!segVol?.colormapLabel} opacity=${segVol?.opacity} ` +
            `name=${segVol?.name}`
        );
        nv.setSliceType(view3d ? nv.sliceTypeRender : nv.sliceTypeMultiplanar);
        nv.setRenderAzimuthElevation(45, 20);
        nv.drawScene();
      };
      nv.attachToCanvas(canvasRef.current);
      nvRef.current = nv;
    }
    const nv = nvRef.current;
    (async () => {
      try {
        const [baseBlob, segBlob] = await Promise.all([
          getCaseViewerVolume(selected.case_id, "t1ce"),
          getCaseViewerVolume(selected.case_id, "seg"),
        ]);
        await nv.loadVolumes([
          { url: URL.createObjectURL(baseBlob), name: "t1ce.nii.gz" },
          { url: URL.createObjectURL(segBlob), name: "seg.nii.gz" },
        ]);
        if (nv.volumes && nv.volumes[1]) nv.volumes[1].setColormapLabel(SEG_CMAP);
        setView3d(true);
        nv.setSliceType(nv.sliceTypeRender);
        nv.setRenderAzimuthElevation(45, 20);
        nv.drawScene();
        setLoadState("done");
      } catch (err: any) {
        setError(err.message || "Failed to load volume");
        setLoadState("error");
      }
    })();
  }, [selected]);

  const toggleView = (to3d: boolean) => {
    const nv = nvRef.current;
    if (!nv) return;
    setView3d(to3d);
    nv.setSliceType(to3d ? nv.sliceTypeRender : nv.sliceTypeMultiplanar);
    nv.drawScene();
  };

  const runSegmentation = async () => {
    const nv = nvRef.current;
    if (!nv || !selected || segState === "running") return;
    setSegState("running");
    setError("");
    try {
      const metrics = await segmentCaseViewerCase(selected.case_id);
      setPredMetrics(metrics);
      const predBlob = await getCaseViewerVolume(selected.case_id, "pred");
      const baseBlob = await getCaseViewerVolume(selected.case_id, "t1ce");
      await nv.loadVolumes([
        { url: URL.createObjectURL(baseBlob), name: "t1ce.nii.gz" },
        { url: URL.createObjectURL(predBlob), name: "seg.nii" },
      ]);
      if (nv.volumes && nv.volumes[1]) nv.volumes[1].setColormapLabel(SEG_CMAP);
      setMaskSource("model");
      nv.setSliceType(view3d ? nv.sliceTypeRender : nv.sliceTypeMultiplanar);
      nv.drawScene();
      setSegState("done");
    } catch (err: any) {
      setError(err.message || "Segmentation failed");
      setSegState("error");
    }
  };

  const switchMask = (src: "expert" | "model") => {
    const nv = nvRef.current;
    if (!nv || !selected || src === maskSource) return;
    if (src === "model" && !predMetrics) {
      runSegmentation();
      return;
    }
    const load = async () => {
      const [baseBlob, maskBlob] = await Promise.all([
        getCaseViewerVolume(selected.case_id, "t1ce"),
        getCaseViewerVolume(selected.case_id, src === "expert" ? "seg" : "pred"),
      ]);
      await nv.loadVolumes([
        { url: URL.createObjectURL(baseBlob), name: "t1ce.nii.gz" },
        { url: URL.createObjectURL(maskBlob), name: "seg.nii" },
      ]);
      if (nv.volumes && nv.volumes[1]) nv.volumes[1].setColormapLabel(SEG_CMAP);
      setMaskSource(src);
      nv.setSliceType(view3d ? nv.sliceTypeRender : nv.sliceTypeMultiplanar);
      nv.drawScene();
    };
    load().catch((err: any) => setError(err.message || "Failed to switch mask"));
  };

  const metrics = maskSource === "model" && predMetrics ? predMetrics.volumes_cm3 : null;

  const pct = (part: number, total: number) =>
    total > 0 ? ((part / total) * 100).toFixed(1) : "0.0";

  return (
    <div className="page-container" style={{ maxWidth: "1400px" }}>
      <h1 className="page-title">
        <Brain size={24} style={{ color: "var(--accent)" }} /> Case Viewer — 3D Tumor Analysis
      </h1>
      <p className="text-muted" style={{ marginBottom: "1.5rem" }}>
        BraTS 2021 sample volumes · multi-region masks (ET · ED · NCR) · real volume in cm³ · kéo chuột để xoay 3D
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

      {selected && (
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
                  <p style={{ color: "#94a3b8", fontSize: "0.85rem" }}>Đang tải volume…</p>
                </div>
              )}
              {loadState === "error" && (
                <div className="loading-screen" style={{ position: "absolute", inset: 0, background: "rgba(15,17,21,0.7)" }}>
                  <p style={{ color: "#f87171", fontSize: "0.85rem" }}>Không tải được volume</p>
                </div>
              )}
            </div>
            <p className="text-muted" style={{ fontSize: "0.75rem", marginTop: "0.5rem", display: "flex", gap: "0.4rem", alignItems: "center" }}>
              <Boxes size={14} /> chuột trái kéo = xoay · scroll = zoom · chuột phải kéo = pan
              {diag && (
                <code style={{ marginLeft: "auto", color: "#64748b", fontSize: "0.7rem" }}>{diag}</code>
              )}
            </p>
          </div>

          <div className="card" style={{ padding: "1rem", background: "#fff" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <p className="eyebrow" style={{ marginTop: 0 }}>TUMOR COMPOSITION</p>
              <div style={{ display: "flex", gap: "0.3rem" }}>
                <button
                  className={`btn btn-tiny ${maskSource === "expert" ? "active" : ""}`}
                  onClick={() => switchMask("expert")}
                >
                  Expert
                </button>
                <button
                  className={`btn btn-tiny ${maskSource === "model" ? "active" : ""}`}
                  onClick={() => switchMask("model")}
                >
                  Model
                </button>
              </div>
            </div>
            {REGION_META.map(({ key, label, color }) => (
              <div key={key} style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
                <span style={{ width: 12, height: 12, borderRadius: 3, background: color }} />
                <span style={{ flex: 1 }}>{label}</span>
                <strong>{(metrics ? metrics[key] : selected.volumes_cm3[key]).toFixed(1)} cm³</strong>
                <span className="text-muted" style={{ width: "3rem", textAlign: "right" }}>
                  {pct(metrics ? metrics[key] : selected.volumes_cm3[key], metrics ? metrics.total : selected.volumes_cm3.total)}%
                </span>
              </div>
            ))}
            <div style={{ borderTop: "1px solid rgba(45,49,66,0.12)", paddingTop: "0.75rem", display: "flex", justifyContent: "space-between" }}>
              <span>Total</span>
              <strong style={{ color: "var(--accent)" }}>
                {(metrics ? metrics.total : selected.volumes_cm3.total).toFixed(1)} cm³
              </strong>
            </div>

            {maskSource === "expert" ? (
              <button className="btn btn-primary btn-block" style={{ marginTop: "1rem" }} onClick={runSegmentation} disabled={segState === "running"}>
                <Wand2 size={16} /> {segState === "running" ? "Đang phân đoạn… ≈2 phút" : "Phân đoạn 3D · SwinUNETR"}
              </button>
            ) : (
              <div className="text-muted" style={{ marginTop: "0.75rem", fontSize: "0.8rem" }}>
                Mask từ model SwinUNETR · {predMetrics ? `${predMetrics.seconds}s GPU` : ""}
              </div>
            )}
            {segState === "error" && maskSource === "expert" && (
              <div className="text-muted" style={{ marginTop: "0.5rem", fontSize: "0.8rem", color: "#ef4444" }}>
                Phân đoạn thất bại — xem console
              </div>
            )}
            <div className="text-muted" style={{ marginTop: "0.75rem", fontSize: "0.8rem", display: "flex", gap: "0.4rem", alignItems: "center" }}>
              <Layers size={14} /> {maskSource === "expert" ? "mask expert seg" : "mask dự đoán (inference ~2 phút, lưu cache)"}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
