import { useRef } from "react";
import { motion, useInView, useScroll, useTransform } from "framer-motion";
import { cn } from "@/lib/utils";

interface PipelineStageProps {
  stageNumber: number;
  title: string;
  fileName: string;
  bullets: string[];
  outputs: string;
  icon: React.ReactNode;
  accentColor?: string;
  visualClassName?: string;
  disableVisualGlow?: boolean;
  reverseLayout?: boolean;
  layoutMode?: "split" | "background";
  backgroundInteractive?: boolean;
  backgroundTextGlass?: boolean;
  backgroundTextDark?: boolean;
  backgroundTextHighContrast?: boolean;
  children?: React.ReactNode;
}

export default function PipelineStage({
  stageNumber,
  title,
  fileName,
  bullets,
  outputs,
  icon,
  visualClassName,
  disableVisualGlow = false,
  reverseLayout = false,
  layoutMode = "split",
  backgroundInteractive = false,
  backgroundTextGlass = false,
  backgroundTextDark = false,
  backgroundTextHighContrast = false,
  children,
}: PipelineStageProps) {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: false, margin: "-20%" });
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start end", "end start"],
  });

  const visualStartX = reverseLayout ? 140 : -140;
  const visualMidOutX = reverseLayout ? 40 : -40;
  const visualEndX = reverseLayout ? 200 : -200;
  const textStartX = reverseLayout ? -80 : 80;
  const textMidOutX = reverseLayout ? -20 : 20;
  const textEndX = reverseLayout ? -90 : 90;

  const visualX = useTransform(
    scrollYProgress,
    [0, 0.2, 0.5, 0.82, 1],
    [visualStartX, 0, 0, visualMidOutX, visualEndX],
  );
  const visualOpacity = useTransform(scrollYProgress, [0, 0.14, 0.82, 1], [0, 1, 1, 0]);
  const visualScale = useTransform(scrollYProgress, [0, 0.2, 0.8, 1], [0.92, 1, 1, 0.94]);

  const textX = useTransform(
    scrollYProgress,
    [0, 0.2, 0.5, 0.82, 1],
    [textStartX, 0, 0, textMidOutX, textEndX],
  );
  const textOpacity = useTransform(scrollYProgress, [0, 0.16, 0.88, 1], [0, 1, 1, 0]);
  const textYBackground = useTransform(scrollYProgress, [0, 0.2, 0.8, 1], [80, 0, 0, -70]);
  const bgScale = useTransform(scrollYProgress, [0, 0.2, 0.8, 1], [1.08, 1, 1.02, 1.12]);
  const bgOpacity = useTransform(scrollYProgress, [0, 0.12, 0.88, 1], [0, 1, 1, 0]);

  return (
    <section
      ref={ref}
      className="relative min-h-screen flex items-center justify-center px-6 py-20 overflow-hidden"
    >
      {/* Connector line */}
      <motion.div
        className="absolute left-1/2 top-0 w-px h-20 bg-gradient-to-b from-transparent to-primary/40"
        initial={{ scaleY: 0 }}
        animate={isInView ? { scaleY: 1 } : { scaleY: 0 }}
        transition={{ duration: 0.8 }}
        style={{ originY: 0 }}
      />

      {layoutMode === "background" ? (
        <>
          <motion.div
            className={cn(
              "absolute inset-0 z-0",
              backgroundInteractive ? "pointer-events-auto" : "pointer-events-none",
            )}
            style={{ opacity: bgOpacity, scale: bgScale }}
          >
            {children}
          </motion.div>

          <div className="relative z-10 max-w-4xl w-full mx-auto px-2 pointer-events-none">
            <motion.div
              className={cn(
                "space-y-6 text-center",
                backgroundTextGlass &&
                  "rounded-2xl p-6 md:p-8 bg-background/8 backdrop-blur-lg border border-foreground/10 shadow-[0_14px_50px_hsl(var(--foreground)/0.08)]",
              )}
              style={{ opacity: textOpacity, y: textYBackground }}
            >
              {/* Stage badge */}
              <motion.div
                className={cn(
                  "inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-sm font-heading",
                  backgroundTextDark
                    ? "bg-black/20 border border-black/20 text-slate-900"
                    : "glass text-primary",
                )}
                initial={{ opacity: 0, y: 20 }}
                animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
                transition={{ duration: 0.5, delay: 0.3 }}
              >
                <span
                  className={cn(
                    "w-2 h-2 rounded-full",
                    backgroundTextDark ? "bg-slate-900" : "bg-primary animate-pulse-glow",
                  )}
                />
                STAGE {stageNumber}
              </motion.div>

              <h2
                className={cn(
                  "text-3xl md:text-5xl font-heading font-bold",
                  backgroundTextDark ? "text-slate-900" : "text-foreground",
                )}
              >
                {title}
              </h2>

              <p
                className={cn(
                  "text-sm font-mono",
                  backgroundTextDark ? "text-slate-700" : "text-muted-foreground",
                  backgroundTextHighContrast && !backgroundTextDark && "text-foreground/85",
                )}
              >
                {fileName}
              </p>

              <ul className="space-y-3 max-w-3xl mx-auto">
                {bullets.map((bullet, i) => (
                  <motion.li
                    key={i}
                    className={cn(
                      "flex items-center justify-center gap-3",
                      backgroundTextDark ? "text-slate-800" : "text-muted-foreground",
                      backgroundTextHighContrast && !backgroundTextDark && "text-foreground/90",
                    )}
                    initial={{ opacity: 0, y: 12 }}
                    animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 12 }}
                    transition={{ duration: 0.5, delay: 0.4 + i * 0.1 }}
                  >
                    <span className={cn("w-1.5 h-1.5 rounded-full shrink-0", backgroundTextDark ? "bg-slate-900" : "bg-primary")} />
                    <span>{bullet}</span>
                  </motion.li>
                ))}
              </ul>

              <motion.div
                className={cn(
                  "rounded-lg p-4 font-mono text-xs max-w-2xl mx-auto",
                  backgroundTextDark
                    ? "bg-black/15 border border-black/20 text-slate-900"
                    : "glass text-primary/80",
                  backgroundTextHighContrast && !backgroundTextDark && "text-foreground",
                )}
                initial={{ opacity: 0, y: 20 }}
                animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
                transition={{ duration: 0.5, delay: 0.7 }}
              >
                <span
                  className={cn(
                    backgroundTextDark ? "text-slate-700" : "text-muted-foreground",
                    backgroundTextHighContrast && !backgroundTextDark && "text-foreground/80",
                  )}
                >
                  Output →
                </span>{" "}
                {outputs}
              </motion.div>
            </motion.div>
          </div>
        </>
      ) : (
        <div className="max-w-6xl w-full mx-auto grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Visual side */}
          <motion.div
            className={cn(
              "relative flex items-center justify-center",
              reverseLayout && "lg:col-start-2",
            )}
            style={{ x: visualX, opacity: visualOpacity, scale: visualScale }}
          >
            <div className={cn("relative w-full max-w-md aspect-square", visualClassName)}>
              {/* Glow background */}
              {!disableVisualGlow && <div className="absolute inset-0 rounded-2xl bg-primary/5 blur-3xl" />}
              {/* Icon / Visual */}
              <div className="relative z-10 w-full h-full flex items-center justify-center">
                {children || (
                  <div className="glass rounded-2xl p-8 box-glow w-full h-full flex items-center justify-center">
                    <div className="text-primary">{icon}</div>
                  </div>
                )}
              </div>
            </div>
          </motion.div>

          {/* Text side */}
          <motion.div
            className={cn("space-y-6", reverseLayout && "lg:col-start-1 lg:row-start-1")}
            style={{ x: textX, opacity: textOpacity }}
          >
          {/* Stage badge */}
          <motion.div
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full glass text-sm font-heading text-primary"
            initial={{ opacity: 0, y: 20 }}
            animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <span className="w-2 h-2 rounded-full bg-primary animate-pulse-glow" />
            STAGE {stageNumber}
          </motion.div>

          <h2 className="text-3xl md:text-4xl font-heading font-bold text-foreground">
            {title}
          </h2>

          <p className="text-sm font-mono text-muted-foreground">{fileName}</p>

          <ul className="space-y-3">
            {bullets.map((bullet, i) => (
              <motion.li
                key={i}
                className="flex items-start gap-3 text-muted-foreground"
                initial={{ opacity: 0, x: 20 }}
                animate={isInView ? { opacity: 1, x: 0 } : { opacity: 0, x: 20 }}
                transition={{ duration: 0.5, delay: 0.4 + i * 0.1 }}
              >
                <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-primary shrink-0" />
                <span>{bullet}</span>
              </motion.li>
            ))}
          </ul>

          {/* Output */}
          <motion.div
            className="glass rounded-lg p-4 font-mono text-xs text-primary/80"
            initial={{ opacity: 0, y: 20 }}
            animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
            transition={{ duration: 0.5, delay: 0.7 }}
          >
            <span className="text-muted-foreground">Output → </span>
            {outputs}
          </motion.div>
          </motion.div>
        </div>
      )}
    </section>
  );
}
