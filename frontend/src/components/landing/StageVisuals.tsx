import { motion, useInView } from "framer-motion";
import Spline from "@splinetool/react-spline";
import { useEffect, useRef, useState } from "react";

export function SplineIngestionVisual() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: false, margin: "-20%" });
  const [shouldRenderSpline, setShouldRenderSpline] = useState(false);

  useEffect(() => {
    if (isInView) {
      setShouldRenderSpline(true);
    }
  }, [isInView]);

  return (
    <div ref={ref} className="glass rounded-2xl box-glow w-full h-full overflow-hidden relative">
      <motion.div
        className="absolute inset-0 bg-[radial-gradient(circle_at_30%_30%,hsl(var(--primary)/0.2),transparent_60%)]"
        initial={{ opacity: 0 }}
        animate={isInView ? { opacity: 1 } : { opacity: 0 }}
        transition={{ duration: 0.8 }}
      />

      <motion.div
        className="relative z-10 w-full h-full min-h-[380px] sm:min-h-[420px] p-3 sm:p-4"
        initial={{ opacity: 0, scale: 0.96 }}
        animate={isInView ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.96 }}
        transition={{ duration: 0.9, ease: "easeOut" }}
      >
        <div className="w-full h-full rounded-xl overflow-hidden bg-[#070d1f]/80 flex items-center justify-center">
          {shouldRenderSpline ? (
            <div className="w-full h-full [transform:scale(0.9)] [transform-origin:center_center]">
              <Spline scene="https://prod.spline.design/CiRL7KnwRkcw-gf6/scene.splinecode" />
            </div>
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <span className="text-xs font-mono text-muted-foreground">Loading 3D scene...</span>
            </div>
          )}
        </div>
      </motion.div>

      <div className="absolute bottom-3 left-3 right-3 rounded-lg border border-primary/15 bg-background/30 backdrop-blur px-3 py-2">
        <p className="text-[10px] font-mono text-primary/80">Centered full-frame Spline visual</p>
      </div>
    </div>
  );
}

export function SplinePreprocessVisual() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: false, margin: "-20%" });
  const [shouldRenderSpline, setShouldRenderSpline] = useState(false);

  useEffect(() => {
    if (isInView) {
      setShouldRenderSpline(true);
    }
  }, [isInView]);

  return (
    <div ref={ref} className="glass rounded-2xl box-glow w-full h-full overflow-hidden relative">
      <motion.div
        className="absolute inset-0 bg-[radial-gradient(circle_at_65%_35%,hsl(var(--secondary)/0.18),transparent_62%)]"
        initial={{ opacity: 0 }}
        animate={isInView ? { opacity: 1 } : { opacity: 0 }}
        transition={{ duration: 0.8 }}
      />

      <motion.div
        className="relative z-10 w-full h-full min-h-[360px] sm:min-h-[410px] p-3 sm:p-4"
        initial={{ opacity: 0, scale: 0.96 }}
        animate={isInView ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.96 }}
        transition={{ duration: 0.9, ease: "easeOut" }}
      >
        <div className="w-full h-full rounded-xl overflow-hidden bg-[#060b19]/85 flex items-center justify-center">
          {shouldRenderSpline ? (
            <div className="w-full h-full [transform:scale(0.9)] [transform-origin:center_center]">
              <Spline scene="https://prod.spline.design/mluotX0vhaRMyBAS/scene.splinecode" />
            </div>
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <span className="text-xs font-mono text-muted-foreground">Loading 3D scene...</span>
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
}

export function SplinePreprocessBackgroundVisual() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: false, margin: "-20%" });
  const [shouldRenderSpline, setShouldRenderSpline] = useState(false);

  useEffect(() => {
    if (isInView) {
      setShouldRenderSpline(true);
    }
  }, [isInView]);

  return (
    <div ref={ref} className="w-full h-full relative overflow-hidden">
      <motion.div
        className="absolute inset-0 overflow-hidden pointer-events-auto z-0"
        initial={{ opacity: 0, scale: 1.03 }}
        animate={isInView ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 1.03 }}
        transition={{ duration: 0.9, ease: "easeOut" }}
      >
        {shouldRenderSpline ? (
          <div className="w-full h-full [transform:scale(3)] [transform-origin:center_center]">
            <Spline scene="https://prod.spline.design/mluotX0vhaRMyBAS/scene.splinecode" />
          </div>
        ) : (
          <div className="w-full h-full bg-[#070a16]" />
        )}
      </motion.div>

      <div className="absolute inset-0 pointer-events-none z-10 bg-[radial-gradient(circle_at_50%_40%,hsl(var(--secondary)/0.16),transparent_70%)]" />
      <div className="absolute inset-0 pointer-events-none z-10 bg-gradient-to-b from-background/5 via-background/10 to-background/25" />
    </div>
  );
}

export function SplineDetectionBackgroundVisual() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: false, margin: "-20%" });
  const [shouldRenderSpline, setShouldRenderSpline] = useState(false);

  useEffect(() => {
    if (isInView) {
      setShouldRenderSpline(true);
    }
  }, [isInView]);

  return (
    <div ref={ref} className="w-full h-full relative overflow-hidden pointer-events-none">
      <motion.div
        className="absolute inset-0 overflow-hidden pointer-events-none z-0"
        initial={{ opacity: 0, scale: 1.03 }}
        animate={isInView ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 1.03 }}
        transition={{ duration: 0.9, ease: "easeOut" }}
      >
        {shouldRenderSpline ? (
          <div className="w-full h-full pointer-events-none [transform:scale(1.22)] [transform-origin:center_center] [filter:saturate(0.7)_brightness(0.55)_contrast(1.05)]">
            <Spline scene="https://prod.spline.design/XLBNexSpNR4za0AV/scene.splinecode" />
          </div>
        ) : (
          <div className="w-full h-full bg-[#070a16]" />
        )}
      </motion.div>

      <div className="absolute inset-0 pointer-events-none z-10 bg-[radial-gradient(circle_at_48%_35%,hsl(var(--primary)/0.16),transparent_68%)]" />
      <div className="absolute inset-0 pointer-events-none z-10 bg-[linear-gradient(120deg,hsl(var(--secondary)/0.14)_0%,transparent_45%,hsl(var(--primary)/0.12)_100%)]" />
      <div className="absolute inset-0 pointer-events-none z-10 bg-gradient-to-b from-background/12 via-background/22 to-background/40" />
    </div>
  );
}

export function SplineDetectionRobotVisual() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: false, margin: "-20%" });
  const [shouldRenderSpline, setShouldRenderSpline] = useState(false);

  useEffect(() => {
    if (isInView) {
      setShouldRenderSpline(true);
    }
  }, [isInView]);

  return (
    <div ref={ref} className="w-full h-full rounded-2xl overflow-hidden relative">
      <motion.div
        className="w-full h-full"
        initial={{ opacity: 0, scale: 0.96 }}
        animate={isInView ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.96 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      >
        {shouldRenderSpline ? (
          <div className="w-full h-full [transform:scale(1.08)] [transform-origin:center_center]">
            <Spline scene="https://prod.spline.design/XLBNexSpNR4za0AV/scene.splinecode" />
          </div>
        ) : (
          <div className="w-full h-full" />
        )}
      </motion.div>
    </div>
  );
}

export function SplineBacktrackBackgroundVisual() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: false, margin: "-20%" });
  const [shouldRenderSpline, setShouldRenderSpline] = useState(false);

  useEffect(() => {
    if (isInView) {
      setShouldRenderSpline(true);
    }
  }, [isInView]);

  return (
    <div ref={ref} className="w-full h-full relative overflow-hidden">
      <motion.div
        className="absolute inset-0 overflow-hidden pointer-events-auto z-0"
        initial={{ opacity: 0, scale: 1.03 }}
        animate={isInView ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 1.03 }}
        transition={{ duration: 0.9, ease: "easeOut" }}
      >
        {shouldRenderSpline ? (
          <div className="w-full h-full [transform:scale(1.18)] [transform-origin:center_center]">
            <Spline scene="https://prod.spline.design/7zDr3Xu9EtlR2Yqp/scene.splinecode" />
          </div>
        ) : (
          <div className="w-full h-full bg-[#070a16]" />
        )}
      </motion.div>

      <div className="absolute inset-0 pointer-events-none z-10 bg-[radial-gradient(circle_at_50%_35%,hsl(var(--secondary)/0.13),transparent_70%)]" />
      <div className="absolute inset-0 pointer-events-none z-10 bg-gradient-to-b from-background/5 via-background/12 to-background/28" />
    </div>
  );
}

export function SplineAttributionBackgroundVisual() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: false, margin: "-20%" });
  const [shouldRenderSpline, setShouldRenderSpline] = useState(false);

  useEffect(() => {
    if (isInView) {
      setShouldRenderSpline(true);
    }
  }, [isInView]);

  return (
    <div ref={ref} className="w-full h-full relative overflow-hidden pointer-events-none">
      <motion.div
        className="absolute inset-0 overflow-hidden pointer-events-none z-0"
        initial={{ opacity: 0, scale: 1.03 }}
        animate={isInView ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 1.03 }}
        transition={{ duration: 0.9, ease: "easeOut" }}
      >
        {shouldRenderSpline ? (
          <div className="w-full h-full pointer-events-none [transform:scale(1.25)] [transform-origin:center_center] [filter:saturate(0.55)_hue-rotate(20deg)_brightness(0.38)_contrast(1.1)]">
            <Spline scene="https://prod.spline.design/hlUJ4ygkS-cjKrNf/scene.splinecode" />
          </div>
        ) : (
          <div className="w-full h-full bg-[#070a16]" />
        )}
      </motion.div>

      <div className="absolute inset-0 pointer-events-none z-10 bg-[radial-gradient(circle_at_50%_38%,hsl(var(--primary)/0.15),transparent_66%)]" />
      <div className="absolute inset-0 pointer-events-none z-10 bg-[linear-gradient(120deg,hsl(var(--primary)/0.2)_0%,transparent_40%,hsl(var(--secondary)/0.18)_100%)]" />
      <div className="absolute inset-0 pointer-events-none z-10 bg-gradient-to-b from-background/34 via-background/48 to-background/70" />
    </div>
  );
}

export function SatelliteBands() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: false, margin: "-20%" });
  const bands = [
    { label: "B02", color: "bg-blue-500" },
    { label: "B03", color: "bg-green-500" },
    { label: "B04", color: "bg-red-500" },
    { label: "B05", color: "bg-orange-400" },
    { label: "B06", color: "bg-yellow-500" },
    { label: "B08", color: "bg-emerald-400" },
    { label: "B11", color: "bg-violet-500" },
    { label: "B12", color: "bg-pink-500" },
  ];

  return (
    <div ref={ref} className="glass rounded-2xl p-8 box-glow w-full h-full flex flex-col items-center justify-center gap-3">
      <div className="text-xs font-mono text-muted-foreground mb-2">Sentinel-2 L2A Bands</div>
      <div className="relative w-48 h-48 perspective-[600px]">
        {bands.map((band, i) => (
          <motion.div
            key={band.label}
            className={`absolute inset-4 rounded-lg ${band.color} opacity-30 border border-foreground/10`}
            initial={{ y: -200, opacity: 0, rotateX: 45 }}
            animate={
              isInView
                ? { y: i * 4, opacity: 0.2 + i * 0.05, rotateX: 0 }
                : { y: -200, opacity: 0, rotateX: 45 }
            }
            transition={{ duration: 0.6, delay: i * 0.12, ease: "easeOut" }}
          />
        ))}
        {bands.map((band, i) => (
          <motion.div
            key={`label-${band.label}`}
            className="absolute right-[-3rem] text-xs font-mono text-muted-foreground"
            style={{ top: `${i * 16 + 16}px` }}
            initial={{ opacity: 0, x: 10 }}
            animate={isInView ? { opacity: 1, x: 0 } : { opacity: 0, x: 10 }}
            transition={{ duration: 0.4, delay: 0.8 + i * 0.08 }}
          >
            {band.label}
          </motion.div>
        ))}
      </div>
    </div>
  );
}

export function PatchGrid() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: false, margin: "-20%" });

  return (
    <div ref={ref} className="glass rounded-2xl p-8 box-glow w-full h-full flex flex-col items-center justify-center">
      <div className="text-xs font-mono text-muted-foreground mb-4">256×256 Patch Tiling</div>
      <div className="grid grid-cols-6 gap-1">
        {Array.from({ length: 36 }).map((_, i) => (
          <motion.div
            key={i}
            className="w-8 h-8 rounded-sm border border-primary/20 bg-primary/5"
            initial={{ scale: 0, opacity: 0 }}
            animate={isInView ? { scale: 1, opacity: 1 } : { scale: 0, opacity: 0 }}
            transition={{
              duration: 0.3,
              delay: i * 0.03,
              ease: "easeOut",
            }}
          >
            <motion.div
              className="w-full h-full bg-primary/10 rounded-sm"
              animate={isInView ? { opacity: [0.1, 0.3, 0.1] } : {}}
              transition={{ duration: 2, repeat: Infinity, delay: i * 0.1 }}
            />
          </motion.div>
        ))}
      </div>
      <div className="flex items-center gap-2 mt-4 text-xs text-muted-foreground">
        <div className="w-3 h-3 border border-dashed border-primary/40 rounded-sm" />
        <span>32px overlap</span>
      </div>
    </div>
  );
}

export function DebrisHeatmap() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: false, margin: "-20%" });

  return (
    <div ref={ref} className="glass rounded-2xl p-8 box-glow w-full h-full flex flex-col items-center justify-center">
      <div className="text-xs font-mono text-muted-foreground mb-4">U-Net Debris Mask</div>
      <div className="relative w-56 h-56">
        {/* Base grid */}
        <div className="absolute inset-0 grid grid-cols-8 gap-px opacity-20">
          {Array.from({ length: 64 }).map((_, i) => (
            <div key={i} className="bg-muted rounded-sm" />
          ))}
        </div>
        {/* Heatmap blobs */}
        {[
          { x: "20%", y: "30%", size: "4rem" },
          { x: "55%", y: "50%", size: "3rem" },
          { x: "35%", y: "70%", size: "2.5rem" },
          { x: "70%", y: "25%", size: "2rem" },
        ].map((blob, i) => (
          <motion.div
            key={i}
            className="absolute rounded-full bg-primary/40 blur-xl"
            style={{ left: blob.x, top: blob.y, width: blob.size, height: blob.size }}
            initial={{ scale: 0, opacity: 0 }}
            animate={isInView ? { scale: 1, opacity: 1 } : { scale: 0, opacity: 0 }}
            transition={{ duration: 0.8, delay: 0.3 + i * 0.2 }}
          />
        ))}
        {/* Detection outlines */}
        {[
          { x: "18%", y: "28%", w: "5rem", h: "4rem" },
          { x: "52%", y: "47%", w: "4rem", h: "3.5rem" },
        ].map((det, i) => (
          <motion.div
            key={`det-${i}`}
            className="absolute border-2 border-primary rounded-lg"
            style={{ left: det.x, top: det.y, width: det.w, height: det.h }}
            initial={{ opacity: 0 }}
            animate={isInView ? { opacity: [0, 1, 0.6, 1] } : { opacity: 0 }}
            transition={{ duration: 2, delay: 1 + i * 0.3, repeat: Infinity }}
          />
        ))}
      </div>
      <div className="flex gap-3 mt-4">
        {["Flip-H", "Flip-V", "Rot90", "Rot180", "Rot270", "Original"].map((tta, i) => (
          <motion.span
            key={tta}
            className="text-[10px] font-mono text-primary/60 px-1.5 py-0.5 rounded border border-primary/20"
            initial={{ opacity: 0, y: 10 }}
            animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 10 }}
            transition={{ delay: 1.5 + i * 0.1 }}
          >
            {tta}
          </motion.span>
        ))}
      </div>
    </div>
  );
}

export function SpectralIndices() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: false, margin: "-20%" });
  const indices = [
    { name: "PI", value: 0.72, color: "bg-primary" },
    { name: "SR", value: 0.58, color: "bg-secondary" },
    { name: "NSI", value: 0.85, color: "bg-primary" },
    { name: "FDI", value: 0.41, color: "bg-secondary" },
  ];
  const polymers = [
    { label: "PE/PP", color: "text-primary border-primary/30" },
    { label: "PET/Nylon", color: "text-secondary border-secondary/30" },
    { label: "Mixed", color: "text-muted-foreground border-muted-foreground/30" },
    { label: "Organic ⚠", color: "text-yellow-400 border-yellow-400/30" },
  ];

  return (
    <div ref={ref} className="glass rounded-2xl p-8 box-glow w-full h-full flex flex-col items-center justify-center gap-6">
      <div className="text-xs font-mono text-muted-foreground">Spectral Index Analysis</div>
      <div className="flex items-end gap-6 h-40">
        {indices.map((idx, i) => (
          <div key={idx.name} className="flex flex-col items-center gap-2">
            <motion.div
              className={`w-6 rounded-t ${idx.color}/60`}
              initial={{ height: 0 }}
              animate={isInView ? { height: `${idx.value * 120}px` } : { height: 0 }}
              transition={{ duration: 0.8, delay: 0.2 + i * 0.15, ease: "easeOut" }}
            />
            <span className="text-xs font-mono text-muted-foreground">{idx.name}</span>
          </div>
        ))}
      </div>
      <div className="flex flex-wrap gap-2 justify-center">
        {polymers.map((p, i) => (
          <motion.span
            key={p.label}
            className={`text-xs font-mono px-3 py-1 rounded-full border ${p.color}`}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={isInView ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.8 }}
            transition={{ delay: 1 + i * 0.15 }}
          >
            {p.label}
          </motion.span>
        ))}
      </div>
    </div>
  );
}

export function BacktrackParticles() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: false, margin: "-20%" });

  const trails = Array.from({ length: 12 }).map((_, i) => ({
    startX: 50 + (Math.random() - 0.5) * 10,
    startY: 50 + (Math.random() - 0.5) * 10,
    endX: Math.random() * 80 + 10,
    endY: Math.random() * 80 + 10,
    delay: i * 0.1,
  }));

  return (
    <div ref={ref} className="glass rounded-2xl p-8 box-glow w-full h-full flex flex-col items-center justify-center">
      <div className="text-xs font-mono text-muted-foreground mb-4">RK4 Backward Integration — 30 days</div>
      <div className="relative w-56 h-56">
        {/* Ocean background */}
        <div className="absolute inset-0 rounded-xl bg-secondary/5 overflow-hidden">
          {/* Current flow lines */}
          {[20, 40, 60, 80].map((y) => (
            <motion.div
              key={y}
              className="absolute h-px bg-secondary/20"
              style={{ top: `${y}%`, left: 0, right: 0 }}
              animate={isInView ? { x: ["-100%", "0%"] } : {}}
              transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
            />
          ))}
        </div>
        {/* Particle trails */}
        <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100">
          {trails.map((trail, i) => (
            <motion.line
              key={i}
              x1={trail.startX}
              y1={trail.startY}
              x2={trail.endX}
              y2={trail.endY}
              stroke="hsl(217, 91%, 60%)"
              strokeWidth="0.5"
              strokeOpacity="0.6"
              initial={{ pathLength: 0 }}
              animate={isInView ? { pathLength: 1 } : { pathLength: 0 }}
              transition={{ duration: 1.5, delay: trail.delay, ease: "easeOut" }}
            />
          ))}
          {/* Cluster endpoints */}
          {[
            { cx: 15, cy: 20 },
            { cx: 80, cy: 75 },
            { cx: 25, cy: 80 },
          ].map((c, i) => (
            <motion.circle
              key={`c-${i}`}
              cx={c.cx}
              cy={c.cy}
              r="3"
              fill="hsl(217, 91%, 60%)"
              fillOpacity="0.4"
              initial={{ r: 0 }}
              animate={isInView ? { r: [0, 5, 3] } : { r: 0 }}
              transition={{ duration: 1, delay: 1.5 + i * 0.2 }}
            />
          ))}
          {/* Detection center */}
          <motion.circle
            cx="50"
            cy="50"
            r="4"
            fill="hsl(166, 72%, 51%)"
            fillOpacity="0.8"
            initial={{ scale: 0 }}
            animate={isInView ? { scale: [0, 1.2, 1] } : { scale: 0 }}
            transition={{ duration: 0.6 }}
          />
        </svg>
      </div>
      <div className="flex gap-4 mt-4 text-[10px] font-mono">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-primary" /> Detection
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-secondary" /> Sources
        </span>
      </div>
    </div>
  );
}

export function AttributionScores() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: false, margin: "-20%" });
  const sources = [
    { label: "Fishing", weight: 0.4, icon: "🎣", color: "bg-primary" },
    { label: "Industrial", weight: 0.3, icon: "🏭", color: "bg-secondary" },
    { label: "Shipping", weight: 0.2, icon: "🚢", color: "bg-primary/70" },
    { label: "River", weight: 0.1, icon: "🌊", color: "bg-secondary/70" },
  ];

  return (
    <div ref={ref} className="glass rounded-2xl p-8 box-glow w-full h-full flex flex-col items-center justify-center gap-6">
      <div className="text-xs font-mono text-muted-foreground">Weighted Source Attribution</div>
      <div className="w-full max-w-xs space-y-4">
        {sources.map((source, i) => (
          <motion.div
            key={source.label}
            className="space-y-1"
            initial={{ opacity: 0, x: -20 }}
            animate={isInView ? { opacity: 1, x: 0 } : { opacity: 0, x: -20 }}
            transition={{ delay: 0.3 + i * 0.15 }}
          >
            <div className="flex items-center justify-between text-sm">
              <span className="flex items-center gap-2">
                <span>{source.icon}</span>
                <span className="text-foreground font-heading">{source.label}</span>
              </span>
              <span className="font-mono text-muted-foreground text-xs">
                {(source.weight * 100).toFixed(0)}%
              </span>
            </div>
            <div className="h-2 rounded-full bg-muted overflow-hidden">
              <motion.div
                className={`h-full rounded-full ${source.color}`}
                initial={{ width: 0 }}
                animate={isInView ? { width: `${source.weight * 100}%` } : { width: 0 }}
                transition={{ duration: 1, delay: 0.5 + i * 0.15, ease: "easeOut" }}
              />
            </div>
          </motion.div>
        ))}
      </div>
      <motion.div
        className="text-xs font-mono text-primary"
        initial={{ opacity: 0 }}
        animate={isInView ? { opacity: [0, 1, 0.7, 1] } : { opacity: 0 }}
        transition={{ duration: 2, delay: 1.5, repeat: Infinity }}
      >
        Confidence: 87%
      </motion.div>
    </div>
  );
}

export function ReportOutput() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: false, margin: "-20%" });
  const outputs = [
    { type: "PDF", desc: "Executive summary, maps, charts", icon: "📄" },
    { type: "GeoJSON", desc: "Merged detections + attribution", icon: "🗺️" },
    { type: "CSV", desc: "Flat cluster table", icon: "📊" },
    { type: "JSON", desc: "Run summary metadata", icon: "⚙️" },
  ];

  return (
    <div ref={ref} className="glass rounded-2xl p-8 box-glow w-full h-full flex flex-col items-center justify-center gap-4">
      <div className="text-xs font-mono text-muted-foreground">Generated Reports</div>
      <div className="grid grid-cols-2 gap-3 w-full max-w-xs">
        {outputs.map((out, i) => (
          <motion.div
            key={out.type}
            className="glass rounded-lg p-4 text-center space-y-2 cursor-default"
            initial={{ opacity: 0, y: 30, scale: 0.9 }}
            animate={isInView ? { opacity: 1, y: 0, scale: 1 } : { opacity: 0, y: 30, scale: 0.9 }}
            transition={{ duration: 0.5, delay: 0.3 + i * 0.15 }}
            whileHover={{ scale: 1.05, borderColor: "hsl(166, 72%, 51%)" }}
          >
            <div className="text-2xl">{out.icon}</div>
            <div className="text-sm font-heading text-foreground">{out.type}</div>
            <div className="text-[10px] text-muted-foreground">{out.desc}</div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
