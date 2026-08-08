import React, { useCallback, useState } from "react";
import { Outlet, useSearchParams } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { ImageViewer } from "./ImageViewer";
import { type PredictionResult } from "../lib/api";

export interface LayoutOutletContext {
  onViewImage: (imageId: number, results: PredictionResult | null) => void;
}

export function DashboardLayout() {
  const [searchParams, setSearchParams] = useSearchParams();
  // In-memory annotations for images opened this session; URL only carries the id.
  const [resultsCache, setResultsCache] = useState<
    Record<number, PredictionResult | null>
  >({});

  const viewParam = searchParams.get("view");
  const viewId = viewParam ? Number(viewParam) : null;

  const handleViewImage = useCallback(
    (imageId: number, results: PredictionResult | null) => {
      setResultsCache((prev) => ({ ...prev, [imageId]: results }));
      setSearchParams({ view: String(imageId) }, { replace: true });
    },
    [setSearchParams],
  );

  const closeViewer = useCallback(() => {
    setSearchParams({}, { replace: true });
    setResultsCache((prev) => {
      if (viewId == null) return prev;
      const next = { ...prev };
      delete next[viewId];
      return next;
    });
  }, [setSearchParams, viewId]);

  return (
    <div className="dashboard-layout">
      <Sidebar />
      <main className="dashboard-content">
        <Outlet context={{ onViewImage: handleViewImage }} />
      </main>
      {viewId != null && (
        <ImageViewer
          imageId={viewId}
          results={resultsCache[viewId] ?? null}
          onClose={closeViewer}
        />
      )}
    </div>
  );
}
