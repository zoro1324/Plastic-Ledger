import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Points, PointMaterial } from "@react-three/drei";
import * as THREE from "three";

function ParticleField() {
  const ref = useRef<THREE.Points>(null);
  const positions = useMemo(() => {
    const pos = new Float32Array(2000 * 3);
    for (let i = 0; i < 2000; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 20;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 20;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 20;
    }
    return pos;
  }, []);

  useFrame((_, delta) => {
    if (ref.current) {
      ref.current.rotation.y += delta * 0.02;
      ref.current.rotation.x += delta * 0.01;
    }
  });

  return (
    <Points ref={ref} positions={positions} stride={3} frustumCulled={false}>
      <PointMaterial transparent color="#2DD4BF" size={0.02} sizeAttenuation depthWrite={false} opacity={0.6} />
    </Points>
  );
}

function WireframeGlobe() {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.15;
    }
  });

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[2, 32, 32]} />
      <meshBasicMaterial color="#2DD4BF" wireframe opacity={0.15} transparent />
    </mesh>
  );
}

function GlobePoints() {
  const ref = useRef<THREE.Points>(null);
  const positions = useMemo(() => {
    const pos: number[] = [];
    const count = 3000;
    for (let i = 0; i < count; i++) {
      const phi = Math.acos(2 * Math.random() - 1);
      const theta = Math.random() * Math.PI * 2;
      const r = 2.01;
      // Create land-like clusters
      const lat = phi - Math.PI / 2;
      const lon = theta - Math.PI;
      const isLand =
        (Math.abs(lat) < 0.8 && Math.abs(lon) < 0.5) ||
        (lat > 0.2 && lat < 1.2 && lon > 0.5 && lon < 2.0) ||
        (lat > -0.5 && lat < 0.3 && lon > -1.5 && lon < -0.3) ||
        (lat > -1.0 && lat < -0.3 && lon > 1.5 && lon < 2.8);
      if (!isLand && Math.random() > 0.3) continue;

      pos.push(
        r * Math.sin(phi) * Math.cos(theta),
        r * Math.cos(phi),
        r * Math.sin(phi) * Math.sin(theta)
      );
    }
    return new Float32Array(pos);
  }, []);

  useFrame((_, delta) => {
    if (ref.current) {
      ref.current.rotation.y += delta * 0.15;
    }
  });

  return (
    <Points ref={ref} positions={positions} stride={3} frustumCulled={false}>
      <PointMaterial transparent color="#2DD4BF" size={0.03} sizeAttenuation depthWrite={false} opacity={0.8} />
    </Points>
  );
}

export default function Globe() {
  return (
    <div className="absolute inset-0 z-0">
      <Canvas camera={{ position: [0, 0, 6], fov: 45 }} dpr={[1, 2]}>
        <ambientLight intensity={0.5} />
        <WireframeGlobe />
        <GlobePoints />
        <ParticleField />
      </Canvas>
    </div>
  );
}
