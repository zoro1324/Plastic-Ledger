import { useEffect, useRef } from "react";
import { Application } from "@splinetool/runtime";

const SPLINE_SCENE_URL = "https://prod.spline.design/KQUi5HWp2z2Rohxm/scene.splinecode";
const HINT_TEXT_QUERY = "move your mouse";

function hideHintNodes(node: unknown) {
  if (!node || typeof node !== "object") return;

  const stack: Array<Record<string, unknown>> = [node as Record<string, unknown>];
  while (stack.length > 0) {
    const current = stack.pop();
    if (!current) continue;

    const name = String(current.name ?? "").toLowerCase();
    if (name.includes(HINT_TEXT_QUERY)) {
      current.visible = false;
    }

    const children = current.children;
    if (Array.isArray(children)) {
      for (const child of children) {
        if (child && typeof child === "object") {
          stack.push(child as Record<string, unknown>);
        }
      }
    }
  }
}

export default function SplineGlobe() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;

    const forwardWheelToPageScroll = (event: WheelEvent) => {
      event.preventDefault();
      window.scrollBy({ top: event.deltaY, left: 0, behavior: "auto" });
    };

    let lastTouchY: number | null = null;
    const onTouchStart = (event: TouchEvent) => {
      if (event.touches.length > 0) {
        lastTouchY = event.touches[0].clientY;
      }
    };

    const onTouchMove = (event: TouchEvent) => {
      if (event.touches.length === 0 || lastTouchY === null) return;
      const currentY = event.touches[0].clientY;
      const deltaY = lastTouchY - currentY;
      lastTouchY = currentY;
      event.preventDefault();
      window.scrollBy({ top: deltaY, left: 0, behavior: "auto" });
    };

    const onTouchEnd = () => {
      lastTouchY = null;
    };

    canvas.addEventListener("wheel", forwardWheelToPageScroll, { passive: false });
    canvas.addEventListener("touchstart", onTouchStart, { passive: true });
    canvas.addEventListener("touchmove", onTouchMove, { passive: false });
    canvas.addEventListener("touchend", onTouchEnd, { passive: true });

    const app = new Application(canvasRef.current);
    void app.load(SPLINE_SCENE_URL).then(() => {
      const runtime = app as Application & {
        findObjectByName?: (name: string) => { visible?: boolean } | undefined;
        _scene?: unknown;
      };

      const exactNames = ["Move your mouse", "Move your mouse.", "move your mouse"];
      for (const name of exactNames) {
        const node = runtime.findObjectByName?.(name);
        if (node) node.visible = false;
      }

      hideHintNodes(runtime._scene);
    });

    return () => {
      canvas.removeEventListener("wheel", forwardWheelToPageScroll);
      canvas.removeEventListener("touchstart", onTouchStart);
      canvas.removeEventListener("touchmove", onTouchMove);
      canvas.removeEventListener("touchend", onTouchEnd);

      const disposable = app as Application & { dispose?: () => void };
      if (typeof disposable.dispose === "function") {
        disposable.dispose();
      }
    };
  }, []);

  return (
    <div className="absolute inset-0 z-0 overflow-hidden" aria-hidden="true">
      <div className="absolute left-1/2 top-1/2 h-[340vh] w-[240vw] -translate-x-[60%] -translate-y-1/2 md:h-[270vh] md:w-[270vw] md:-translate-x-[57%] lg:h-[300vh] lg:w-[300vw] lg:-translate-x-[58%]">
        <canvas
          ref={canvasRef}
          className="h-full w-full"
          style={{ transform: "scale(1.5)", transformOrigin: "center center", filter: "brightness(2)" }}
        />
      </div>
    </div>
  );
}