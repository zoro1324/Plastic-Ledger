import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "@/components/Navbar";
import LandingPage from "@/pages/LandingPage";
import TrackingPage from "@/pages/TrackingPage";
import DashboardPage from "@/pages/DashboardPage";
import DetectionMapPage from "@/pages/DetectionMapPage";
import AttributionPage from "@/pages/AttributionPage";
import AnalyticsPage from "@/pages/AnalyticsPage";
import ReportsPage from "@/pages/ReportsPage";
import RunDetailPage from "@/pages/RunDetailPage";

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-background text-foreground">
        <Navbar />
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/tracking" element={<TrackingPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/detection" element={<DetectionMapPage />} />
          <Route path="/attribution" element={<AttributionPage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
          <Route path="/reports" element={<ReportsPage />} />
          <Route path="/run_details" element={<RunDetailPage />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
};

export default App;
