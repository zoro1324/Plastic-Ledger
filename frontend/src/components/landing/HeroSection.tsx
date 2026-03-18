import { motion } from "framer-motion";
import { Suspense, lazy } from "react";
import { ChevronDown } from "lucide-react";

const SplineGlobe = lazy(() => import("./SplineGlobe"));

export default function HeroSection() {
  const scrollToContent = () => {
    document.getElementById("pipeline-input")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Top Right Dashboard Button */}
      <motion.div 
        className="absolute top-6 right-6 lg:top-8 lg:right-10 z-50 pointer-events-auto"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8, duration: 0.6 }}
      >
        <motion.a
          href="#"
          className="inline-flex items-center justify-center gap-2 px-6 py-2.5 rounded-full bg-primary text-primary-foreground font-heading font-semibold text-sm
                     hover:shadow-[0_0_30px_hsl(166,72%,51%,0.4)] transition-shadow duration-300"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.98 }}
        >
          Go to Dashboard →
        </motion.a>
      </motion.div>

      {/* Spline globe background */}
      <Suspense fallback={<div className="absolute inset-0 bg-background" />}>
        <SplineGlobe />
      </Suspense>

      {/* Gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-background/30 via-background/60 to-background z-10 pointer-events-none" />

      {/* Content */}
      <div className="relative z-20 text-center px-6 space-y-8 pointer-events-none">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, ease: "easeOut" }}
          className="space-y-4"
        >
          <motion.div
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full glass text-xs font-mono text-primary"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3, duration: 0.6 }}
          >
            <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse-glow" />
            SENTINEL-2 POWERED
          </motion.div>

          <h1 className="text-5xl md:text-7xl lg:text-8xl font-heading font-bold tracking-tight text-foreground">
            PLASTIC
            <span className="text-glow text-primary">-</span>
            LEDGER
          </h1>

          <p className="max-w-xl mx-auto text-lg md:text-xl text-muted-foreground font-light">
            Detect, classify, and trace marine microplastics back to their source — from space.
          </p>
        </motion.div>

        <motion.button
          onClick={scrollToContent}
          className="group inline-flex items-center gap-2 px-8 py-3 rounded-full bg-primary text-primary-foreground font-heading font-semibold text-sm
                     hover:shadow-[0_0_30px_hsl(166,72%,51%,0.4)] transition-shadow duration-300 pointer-events-auto"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.6 }}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.98 }}
        >
          Start Pipeline
          <ChevronDown className="w-4 h-4 group-hover:translate-y-0.5 transition-transform" />
        </motion.button>

        {/* Scroll indicator */}
        <motion.div
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <ChevronDown className="w-5 h-5 text-muted-foreground" />
        </motion.div>
      </div>
    </section>
  );
}
