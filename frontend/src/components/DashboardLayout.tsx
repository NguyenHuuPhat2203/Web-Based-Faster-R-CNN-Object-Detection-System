import React, { useState, useCallback } from "react";
import { Sidebar, type DashboardPage } from "./Sidebar";
import { DashboardHome } from "./DashboardHome";
import { Detector } from "./Detector";
import { History } from "./History";
import { Settings } from "./Settings";
import { ImageViewer } from "./ImageViewer";
import { type PredictionResult } from "../lib/api";

export function DashboardLayout() {
  const [page, setPage] = useState<DashboardPage>("home");
  const [viewerImage, setViewerImage] = useState<{
    id: number;
    results: PredictionResult | null;
  } | null>(null);

  const handleViewImage = useCallback(
    (imageId: number, results: PredictionResult | null) => {
      setViewerImage({ id: imageId, results });
    },
    [],
  );

  const closeViewer = useCallback(() => {
    setViewerImage(null);
  }, []);

  const renderPage = () => {
    switch (page) {
      case "home":
        return <DashboardHome onNavigate={setPage} onViewImage={handleViewImage} />;
      case "detector":
        return <Detector />;
      case "history":
        return <History onNavigate={setPage} onViewImage={handleViewImage} />;
      case "settings":
        return <Settings />;
    }
  };

  return (
    <div className="dashboard-layout">
      <Sidebar activePage={page} onNavigate={setPage} />
      <main className="dashboard-content">
        {renderPage()}
      </main>
      {viewerImage && (
        <ImageViewer
          imageId={viewerImage.id}
          results={viewerImage.results}
          onClose={closeViewer}
        />
      )}
    </div>
  );
}
