import React, { useState, useCallback } from "react";
import { Sidebar, type DashboardPage } from "./Sidebar";
import { DashboardHome } from "./DashboardHome";
import { Detector } from "./Detector";
import { History } from "./History";
import { Settings } from "./Settings";
import { type PredictionResult } from "../lib/api";

export function DashboardLayout() {
  const [page, setPage] = useState<DashboardPage>("home");
  const [viewingImage, setViewingImage] = useState<{
    id: number;
    results: PredictionResult | null;
  } | null>(null);

  const handleViewImage = useCallback(
    (imageId: number, results: PredictionResult | null) => {
      setViewingImage({ id: imageId, results });
      setPage("detector");
    },
    [],
  );

  const renderPage = () => {
    switch (page) {
      case "home":
        return <DashboardHome onNavigate={setPage} onViewImage={handleViewImage} />;
      case "detector":
        return (
          <Detector
            key={viewingImage?.id ?? "new"}
            initialImageId={viewingImage?.id}
            initialResults={viewingImage?.results ?? null}
          />
        );
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
    </div>
  );
}
