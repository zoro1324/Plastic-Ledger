import { useRef } from "react";
import { motion, useInView } from "framer-motion";

interface PipelineStageProps {
  stageNumber: number;
  title: string;
  fileName: string;
  bullets: string[];
  outputs: string;
  icon: React.ReactNode;
  accentColor?: string;
  children?: React.ReactNode;
}

export default function PipelineStage({
  stageNumber,
  title,
  fileName,
  bullets,
  outputs,
  icon,
  children,
}: PipelineStageProps) {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: false, margin: "-20%" });

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

      <div className="max-w-6xl w-full mx-auto grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
        {/* Visual side */}
        <motion.div
          className="relative flex items-center justify-center"
          initial={{ opacity: 0, x: -60 }}
          animate={isInView ? { opacity: 1, x: 0 } : { opacity: 0, x: -60 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          <div className="relative w-full max-w-md aspect-square">
            {/* Glow background */}
            <div className="absolute inset-0 rounded-2xl bg-primary/5 blur-3xl" />
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
          className="space-y-6"
          initial={{ opacity: 0, x: 60 }}
          animate={isInView ? { opacity: 1, x: 0 } : { opacity: 0, x: 60 }}
          transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
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
    </section>
  );
}
