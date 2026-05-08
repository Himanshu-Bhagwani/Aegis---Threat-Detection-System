"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

export interface GlobePing {
  lat: number;
  lon: number;
  risk: "minimal" | "low" | "medium" | "high" | "critical";
  label?: string;
}

interface Props {
  pings?: GlobePing[];
  width?: number;
  height?: number;
}

const RISK_COLORS: Record<string, string> = {
  minimal:  "#22c55e",
  low:      "#84cc16",
  medium:   "#f59e0b",
  high:     "#f97316",
  critical: "#ef4444",
};

function latLonToVec3(lat: number, lon: number, r: number): THREE.Vector3 {
  const phi   = (90 - lat)  * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);
  return new THREE.Vector3(
    -(r * Math.sin(phi) * Math.cos(theta)),
     (r * Math.cos(phi)),
     (r * Math.sin(phi) * Math.sin(theta)),
  );
}

export default function ThreatGlobe({ pings = [], width = 420, height = 420 }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!mountRef.current) return;
    const el = mountRef.current;

    /* ── Renderer ─────────────────────────────────────── */
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    el.appendChild(renderer.domElement);

    const scene  = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 0, 2.8);

    /* ── Globe ─────────────────────────────────────────── */
    const globeGeo = new THREE.SphereGeometry(1, 48, 48);
    const globeMat = new THREE.MeshPhongMaterial({
      color: 0x0e1016, emissive: 0x0a0f1a,
      specular: 0x3d7fff, shininess: 40,
      transparent: true, opacity: 0.95,
    });
    const globe = new THREE.Mesh(globeGeo, globeMat);
    scene.add(globe);

    /* ── Wireframe ────────────────────────────────────── */
    const wireGeo = new THREE.SphereGeometry(1.002, 24, 24);
    const wireMesh = new THREE.Mesh(wireGeo,
      new THREE.MeshBasicMaterial({ color: 0x1a2a4a, wireframe: true, transparent: true, opacity: 0.18 })
    );
    scene.add(wireMesh);

    /* ── Atmosphere ───────────────────────────────────── */
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1.08, 32, 32),
      new THREE.MeshBasicMaterial({ color: 0x3d7fff, transparent: true, opacity: 0.05, side: THREE.BackSide })
    ));

    /* ── Lights ───────────────────────────────────────── */
    scene.add(new THREE.AmbientLight(0xffffff, 0.4));
    const dirLight = new THREE.DirectionalLight(0x6699ff, 1.2);
    dirLight.position.set(5, 3, 5);
    scene.add(dirLight);

    /* ── Pings ────────────────────────────────────────── */
    interface PingEntry { base: THREE.Vector3; dot: THREE.Mesh; ring: THREE.Mesh; phase: number }
    const pingEntries: PingEntry[] = pings.map(ping => {
      const base  = latLonToVec3(ping.lat, ping.lon, 1.01);
      const colorStr = RISK_COLORS[ping.risk] ?? "#3d7fff";
      const colorHex = parseInt(colorStr.replace('#', ''), 16);

      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(0.018, 8, 8),
        new THREE.MeshBasicMaterial({ color: colorHex })
      );
      dot.position.copy(base);
      scene.add(dot);

      const ringMat = new THREE.MeshBasicMaterial({
        color: colorHex, transparent: true, opacity: 0.7, side: THREE.DoubleSide,
      });
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.022, 0.030, 16), ringMat);
      ring.position.copy(base);
      ring.lookAt(0, 0, 0);
      scene.add(ring);

      return { base, dot, ring, phase: Math.random() * Math.PI * 2 };
    });

    /* ── Drag ─────────────────────────────────────────── */
    let rotY = 0, isDragging = false, lastX = 0, vel = 0;
    const onDown  = (e: MouseEvent) => { isDragging = true;  lastX = e.clientX; };
    const onUp    = ()              => { isDragging = false; };
    const onMove  = (e: MouseEvent) => {
      if (!isDragging) return;
      vel = (e.clientX - lastX) * 0.008;
      rotY += vel; lastX = e.clientX;
    };
    el.addEventListener("mousedown", onDown);
    window.addEventListener("mouseup",   onUp);
    window.addEventListener("mousemove", onMove);

    /* ── Animate ──────────────────────────────────────── */
    let t = 0, animId = 0;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      t += 0.016;

      if (!isDragging) { vel *= 0.95; rotY += 0.003 + vel; }
      globe.rotation.y   = rotY;
      wireMesh.rotation.y = rotY;

      pingEntries.forEach(({ base, dot, ring, phase }) => {
        // rotate around Y
        const px = base.x * Math.cos(rotY) - base.z * Math.sin(rotY);
        const pz = base.x * Math.sin(rotY) + base.z * Math.cos(rotY);
        dot.position.set(px, base.y, pz);
        ring.position.copy(dot.position);
        ring.lookAt(camera.position);
        const s = 1 + 0.6 * Math.abs(Math.sin(t * 2 + phase));
        ring.scale.setScalar(s);
        (ring.material as THREE.MeshBasicMaterial).opacity = 0.7 * (1 - 0.5 * Math.abs(Math.sin(t * 2 + phase)));
      });

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animId);
      el.removeEventListener("mousedown", onDown);
      window.removeEventListener("mouseup",   onUp);
      window.removeEventListener("mousemove", onMove);
      renderer.dispose();
      if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pings.length, width, height]);

  return (
    <div ref={mountRef} style={{
      width, height, cursor: "grab",
      userSelect: "none", borderRadius: "50%", overflow: "hidden",
    }} />
  );
}
