import React, { useState } from "react";
import { Sidebar, type DashboardPage } from "./Sidebar";
import { DashboardHome } from "./DashboardHome";
import { Detector } from "./Detector";
import { History } from "./History";
import { Settings } from "./Settings";

export function DashboardLayout() {
  const [page, setPage] = useState<DashboardPage>("home");

  const renderPage = () => {
    switch (page) {
      case "home":
        return <DashboardHome onNavigate={setPage} />;
      case "detector":
        return <Detector />;
      case "history":
        return <History onNavigate={setPage} />;
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
