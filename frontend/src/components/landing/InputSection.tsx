import { motion, useInView } from "framer-motion";
import { useRef } from "react";

export default function InputSection() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: false, margin: "-20%" });

  return (
    <section
      id="pipeline-input"
      ref={ref}
      className="relative min-h-[70vh] flex items-center justify-center px-6 py-20"
    >
      <div className="max-w-3xl w-full mx-auto text-center space-y-8">
        <motion.div
          className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full glass text-xs font-mono text-primary"
          initial={{ opacity: 0 }}
          animate={isInView ? { opacity: 1 } : { opacity: 0 }}
          transition={{ duration: 0.5 }}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-primary" />
          PIPELINE INPUT
        </motion.div>

        <motion.h2
          className="text-3xl md:text-4xl font-heading font-bold text-foreground"
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          Define Your Region of Interest
        </motion.h2>

        {/* Bounding box visualization */}
        <motion.div
          className="glass rounded-xl p-6 max-w-md mx-auto space-y-4"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={isInView ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.95 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <div className="relative h-40 rounded-lg bg-muted/30 overflow-hidden">
            {/* Grid lines */}
            <div className="absolute inset-0 grid grid-cols-6 grid-rows-4">
              {Array.from({ length: 24 }).map((_, i) => (
                <div key={i} className="border border-muted/20" />
              ))}
            </div>
            {/* Bbox overlay */}
            <motion.div
              className="absolute border-2 border-primary rounded bg-primary/10"
              style={{ left: "25%", top: "20%", width: "50%", height: "60%" }}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={isInView ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.8 }}
              transition={{ duration: 0.8, delay: 0.8 }}
            >
              {/* Corner markers */}
              {[
                "top-0 left-0 -translate-x-1 -translate-y-1",
                "top-0 right-0 translate-x-1 -translate-y-1",
                "bottom-0 left-0 -translate-x-1 translate-y-1",
                "bottom-0 right-0 translate-x-1 translate-y-1",
              ].map((pos, i) => (
                <div
                  key={i}
                  className={`absolute ${pos} w-2 h-2 bg-primary rounded-full`}
                />
              ))}
            </motion.div>
          </div>

          {/* Coordinates */}
          <div className="grid grid-cols-2 gap-2 text-xs font-mono">
            {[
              { label: "lon_min", value: "23.4521" },
              { label: "lat_min", value: "37.8934" },
              { label: "lon_max", value: "24.1087" },
              { label: "lat_max", value: "38.2156" },
            ].map((coord, i) => (
              <motion.div
                key={coord.label}
                className="flex justify-between glass rounded px-3 py-2"
                initial={{ opacity: 0, x: i % 2 === 0 ? -10 : 10 }}
                animate={isInView ? { opacity: 1, x: 0 } : { opacity: 0, x: i % 2 === 0 ? -10 : 10 }}
                transition={{ delay: 1 + i * 0.1 }}
              >
                <span className="text-muted-foreground">{coord.label}</span>
                <span className="text-primary">{coord.value}</span>
              </motion.div>
            ))}
          </div>

          <motion.div
            className="flex items-center justify-center gap-2 text-xs font-mono glass rounded px-3 py-2"
            initial={{ opacity: 0 }}
            animate={isInView ? { opacity: 1 } : { opacity: 0 }}
            transition={{ delay: 1.5 }}
          >
            <span className="text-muted-foreground">target_date</span>
            <span className="text-primary">2024-06-15</span>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}
