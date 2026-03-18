import { Satellite, Layers, Brain, FlaskConical, Navigation, Target, FileText } from "lucide-react";
import HeroSection from "@/components/landing/HeroSection";
import InputSection from "@/components/landing/InputSection";
import PipelineStage from "@/components/landing/PipelineStage";
import {
  SplineIngestionVisual,
  SplinePreprocessBackgroundVisual,
  DebrisHeatmap,
  SpectralIndices,
  BacktrackParticles,
  AttributionScores,
  ReportOutput,
} from "@/components/landing/StageVisuals";
import { motion, useInView } from "framer-motion";
import { useRef } from "react";

const stages = [
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
    visualClassName: "max-w-lg xl:max-w-xl aspect-[1.08/1]",
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
    Visual: DebrisHeatmap,
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
    Visual: BacktrackParticles,
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
    Visual: AttributionScores,
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
  },
];

function FooterCTA() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: false, margin: "-20%" });

  return (
    <section ref={ref} className="relative min-h-[60vh] flex items-center justify-center px-6 py-20">
      <div className="absolute inset-0 bg-gradient-to-t from-primary/5 to-transparent" />
      <motion.div
        className="relative z-10 text-center space-y-6 max-w-2xl"
        initial={{ opacity: 0, y: 30 }}
        animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
        transition={{ duration: 0.8 }}
      >
        <h2 className="text-4xl md:text-5xl font-heading font-bold text-foreground">
          Pipeline Complete
        </h2>
        <p className="text-muted-foreground text-lg">
          From satellite imagery to actionable intelligence — all outputs ready in your dashboard.
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <motion.a
            href="#"
            className="inline-flex items-center justify-center gap-2 px-8 py-3 rounded-full bg-primary text-primary-foreground font-heading font-semibold text-sm
                       hover:shadow-[0_0_30px_hsl(166,72%,51%,0.4)] transition-shadow duration-300"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.98 }}
          >
            Go to Dashboard →
          </motion.a>
          <motion.a
            href="#"
            className="inline-flex items-center justify-center gap-2 px-8 py-3 rounded-full glass text-foreground font-heading font-semibold text-sm
                       hover:border-primary/50 transition-colors duration-300"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.98 }}
          >
            View Documentation
          </motion.a>
        </div>

        {/* Output file tree */}
        <motion.div
          className="glass rounded-xl p-4 max-w-xs mx-auto text-left font-mono text-xs text-muted-foreground mt-8"
          initial={{ opacity: 0 }}
          animate={isInView ? { opacity: 1 } : { opacity: 0 }}
          transition={{ delay: 0.5 }}
        >
          <div className="text-primary mb-2">data/reports/&lt;SCENE&gt;/</div>
          <div className="pl-4 space-y-1">
            <div>├── final_report.pdf</div>
            <div>├── final_report.geojson</div>
            <div>├── debris_summary.csv</div>
            <div>└── run_summary.json</div>
          </div>
        </motion.div>
      </motion.div>
    </section>
  );
}

export default function Index() {
  return (
    <main className="bg-background min-h-screen overflow-x-hidden">
      <HeroSection />
      <InputSection />

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
          reverseLayout={stage.reverseLayout}
          layoutMode={stage.layoutMode}
          backgroundInteractive={stage.backgroundInteractive}
        >
          <stage.Visual />
        </PipelineStage>
      ))}

      <FooterCTA />

      {/* Footer */}
      <footer className="border-t border-border py-8 px-6 text-center">
        <p className="text-xs text-muted-foreground font-mono">
          PLASTIC-LEDGER — Sentinel-2 Microplastic Detection Pipeline
        </p>
      </footer>
    </main>
  );
}
