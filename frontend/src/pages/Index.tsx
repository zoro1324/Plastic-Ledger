import { Satellite, Layers, Brain, FlaskConical, Navigation, Target, FileText } from "lucide-react";
import HeroSection from "@/components/landing/HeroSection";
import PipelineStage from "@/components/landing/PipelineStage";
import {
  SplineIngestionVisual,
  SplinePreprocessBackgroundVisual,
  SplineDetectionRobotVisual,
  SplineBacktrackBackgroundVisual,
  SplineAttributionBackgroundVisual,
  SpectralIndices,
  ReportOutput,
} from "@/components/landing/StageVisuals";
import { motion, useInView } from "framer-motion";
import { useRef } from "react";

type PipelineStageData = {
  stageNumber: number;
  title: string;
  fileName: string;
  bullets: string[];
  outputs: string;
  icon: React.ReactNode;
  Visual: React.ElementType;
  visualClassName?: string;
  disableVisualGlow?: boolean;
  reverseLayout?: boolean;
  layoutMode?: "split" | "background";
  backgroundInteractive?: boolean;
  backgroundTextGlass?: boolean;
  backgroundTextDark?: boolean;
  backgroundTextHighContrast?: boolean;
};

const stages: PipelineStageData[] = [
  {
    stageNumber: 1,
    title: "Satellite Data Ingestion",
    fileName: "pipeline/01_ingest.py",
    bullets: [
      "Copernicus STAC API → find best Sentinel-2 L2A scene",
      "OpenID Connect auth → download 8 band GeoTIFFs",
      "Store raw bands with full metadata for reproducibility",
    ],
    outputs: "data/raw/<SCENE_ID>/{B02,B04,B08...}.tif + metadata.json",
    icon: <Satellite className="w-20 h-20" />,
    Visual: SplineIngestionVisual,
    layoutMode: "background",
    backgroundInteractive: true,
    backgroundTextGlass: true,
  },
  {
    stageNumber: 2,
    title: "Preprocessing",
    fileName: "pipeline/02_preprocess.py",
    bullets: [
      "8 bands → 11-band model order (zero-pad B01, B06, B07)",
      "Clip [0.0001, 0.5] → Z-score normalise using MARIDA statistics",
      "Tile into 256×256 patches with 32-pixel overlap",
    ],
    outputs: "data/processed/<SCENE>/patches/patch_NNNN.npz",
    icon: <Layers className="w-20 h-20" />,
    Visual: SplinePreprocessBackgroundVisual,
    layoutMode: "background",
    backgroundInteractive: true,
    backgroundTextGlass: true,
  },
  {
    stageNumber: 3,
    title: "Marine Debris Detection",
    fileName: "pipeline/03_detect.py",
    bullets: [
      "U-Net (ResNet-34) trained on MARIDA → 15-class segmentation",
      "Test-Time Augmentation (6 flips/rotations) → averaged probabilities",
      "Patch stitching with overlap averaging → debris mask → connected components",
    ],
    outputs: "detections.geojson + debris_mask.tif + debris_prob.tif",
    icon: <Brain className="w-20 h-20" />,
    Visual: SplineDetectionRobotVisual,
    disableVisualGlow: true,
  },
  {
    stageNumber: 4,
    title: "Polymer Type Classification",
    fileName: "pipeline/04_polymer.py",
    bullets: [
      "Extract mean 11-band spectrum per cluster",
      "Compute PI, SR, NSI, FDI spectral indices",
      "Rule-based decision tree → PE/PP, PET/Nylon, Mixed, Organic",
    ],
    outputs: "detections_classified.geojson",
    icon: <FlaskConical className="w-20 h-20" />,
    Visual: SpectralIndices,
    layoutMode: "background",
    backgroundInteractive: true,
    backgroundTextHighContrast: true,
  },
  {
    stageNumber: 5,
    title: "Hydrodynamic Back-Tracking",
    fileName: "pipeline/05_backtrack.py",
    bullets: [
      "Download CMEMS ocean currents (uo, vo) + ERA5 wind data",
      "Release 50 particles per cluster, RK4 integration backward 30 days",
      "DBSCAN-cluster trajectory endpoints → source regions",
    ],
    outputs: "backtrack_summary.json + trajectory GeoJSONs",
    icon: <Navigation className="w-20 h-20" />,
    Visual: SplineBacktrackBackgroundVisual,
    layoutMode: "background",
    backgroundInteractive: true,
    backgroundTextGlass: true,
    backgroundTextDark: true,
  },
  {
    stageNumber: 6,
    title: "Source Attribution",
    fileName: "pipeline/06_attribute.py",
    bullets: [
      "Fishing: GFW API vessel activity (weight 0.40)",
      "Industrial: OSM waste sites (weight 0.30)",
      "Shipping: Shipping lane overlap (weight 0.20)",
      "River: Distance to river mouths (weight 0.10)",
    ],
    outputs: "attribution_report.json",
    icon: <Target className="w-20 h-20" />,
    Visual: SplineAttributionBackgroundVisual,
    layoutMode: "background",
    backgroundInteractive: false,
    backgroundTextGlass: true,
    backgroundTextHighContrast: true,
  },
  {
    stageNumber: 7,
    title: "Report Generation",
    fileName: "pipeline/07_report.py",
    bullets: [
      "PDF: executive summary, maps, polymer chart, cluster table",
      "GeoJSON: merged detections + attribution",
      "CSV: flat cluster table for data analysis",
    ],
    outputs: "final_report.pdf + final_report.geojson + debris_summary.csv",
    icon: <FileText className="w-20 h-20" />,
    Visual: ReportOutput,
    layoutMode: "background",
    backgroundInteractive: true,
    backgroundTextGlass: true,
    backgroundTextHighContrast: true,
  },
];



export default function Index() {
  return (
    <main className="bg-background min-h-screen overflow-x-hidden">
      <HeroSection />

      {/* Pipeline vertical connector */}
      <div className="flex justify-center">
        <div className="w-px h-20 bg-gradient-to-b from-transparent via-primary/30 to-transparent" />
      </div>

      {stages.map((stage) => (
        <PipelineStage
          key={stage.stageNumber}
          stageNumber={stage.stageNumber}
          title={stage.title}
          fileName={stage.fileName}
          bullets={stage.bullets}
          outputs={stage.outputs}
          icon={stage.icon}
          visualClassName={stage.visualClassName}
          disableVisualGlow={stage.disableVisualGlow}
          reverseLayout={stage.reverseLayout}
          layoutMode={stage.layoutMode}
          backgroundInteractive={stage.backgroundInteractive}
          backgroundTextGlass={stage.backgroundTextGlass}
          backgroundTextDark={stage.backgroundTextDark}
          backgroundTextHighContrast={stage.backgroundTextHighContrast}
        >
          <stage.Visual />
        </PipelineStage>
      ))}



      {/* Footer */}
      <footer className="border-t border-border py-8 px-6 text-center">
        <p className="text-xs text-muted-foreground font-mono">
          PLASTIC-LEDGER — Sentinel-2 Microplastic Detection Pipeline
        </p>
      </footer>
    </main>
  );
}
