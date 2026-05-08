"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

interface Props { score: number; size?: number }

function scoreToColor(score: number): number {
  if (score >= 0.75) return 0xef4444;
  if (score >= 0.50) return 0xf97316;
  if (score >= 0.25) return 0xf59e0b;
  if (score >= 0.10) return 0x84cc16;
  return 0x22c55e;
}

export default function RiskGauge3D({ score, size = 240 }: Props) {
  const mountRef  = useRef<HTMLDivElement>(null);
  const scoreRef  = useRef(score);
  const currentRef = useRef(0);
  scoreRef.current = score;

  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(size, size);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    el.appendChild(renderer.domElement);

    const scene  = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
    camera.position.set(0, 0, 4);

    /* ── Ambient glow circle ──────────────────────────── */
    const glowGeo = new THREE.RingGeometry(1.28, 1.38, 64);
    const glowMat = new THREE.MeshBasicMaterial({
      color: 0x3d7fff, transparent: true, opacity: 0.06, side: THREE.DoubleSide,
    });
    scene.add(new THREE.Mesh(glowGeo, glowMat));

    /* ── Track torus ──────────────────────────────────── */
    const TRACK_R = 1.2, TUBE_R = 0.055;
    const trackGeo = new THREE.TorusGeometry(TRACK_R, TUBE_R, 16, 120);
    const trackMat = new THREE.MeshBasicMaterial({ color: 0x1a1f2e });
    scene.add(new THREE.Mesh(trackGeo, trackMat));

    /* ── Fill torus (arc from bottom, clockwise) ──────── */
    const SEGS   = 120;
    const fillGeo = new THREE.TorusGeometry(TRACK_R, TUBE_R * 1.1, 16, SEGS, 0);
    const fillMat = new THREE.MeshBasicMaterial({ color: 0x22c55e });
    const fillMesh = new THREE.Mesh(fillGeo, fillMat);
    fillMesh.rotation.z = -Math.PI / 2; // start at top
    scene.add(fillMesh);

    /* ── Centre sphere ────────────────────────────────── */
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(0.78, 32, 32),
      new THREE.MeshBasicMaterial({ color: 0x0a0c14 })
    ));

    /* ── Tick marks ───────────────────────────────────── */
    for (let i = 0; i < 12; i++) {
      const angle = (i / 12) * Math.PI * 2;
      const inner = 1.08, outer = 1.16;
      const points = [
        new THREE.Vector3(Math.cos(angle) * inner, Math.sin(angle) * inner, 0),
        new THREE.Vector3(Math.cos(angle) * outer, Math.sin(angle) * outer, 0),
      ];
      const lGeo = new THREE.BufferGeometry().setFromPoints(points);
      scene.add(new THREE.Line(lGeo, new THREE.LineBasicMaterial({ color: 0x1e2535, transparent: true, opacity: 0.6 })));
    }

    /* ── Lights ───────────────────────────────────────── */
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dl = new THREE.DirectionalLight(0xffffff, 0.8);
    dl.position.set(2, 3, 4);
    scene.add(dl);

    /* ── Animate ──────────────────────────────────────── */
    let animId = 0;
    const animate = () => {
      animId = requestAnimationFrame(animate);

      // lerp toward target
      const target = scoreRef.current;
      currentRef.current += (target - currentRef.current) * 0.04;
      const s = currentRef.current;

      // rebuild fill arc geometry
      const newFill = new THREE.TorusGeometry(TRACK_R, TUBE_R * 1.1, 16, SEGS, s * Math.PI * 2);
      fillMesh.geometry.dispose();
      fillMesh.geometry = newFill;
      (fillMesh.material as THREE.MeshBasicMaterial).color.setHex(scoreToColor(s));

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animId);
      renderer.dispose();
      if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement);
    };
  }, [size]);

  const pct   = Math.round(score * 100);
  const label = score >= 0.75 ? "CRITICAL" : score >= 0.5 ? "HIGH" : score >= 0.25 ? "MEDIUM" : score >= 0.1 ? "LOW" : "MINIMAL";
  const hex   = ["#22c55e","#84cc16","#f59e0b","#f97316","#ef4444"][
    score >= 0.75 ? 4 : score >= 0.5 ? 3 : score >= 0.25 ? 2 : score >= 0.1 ? 1 : 0
  ];

  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <div ref={mountRef} />
      {/* Overlay text */}
      <div style={{
        position: "absolute", inset: 0,
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        pointerEvents: "none",
      }}>
        <div style={{
          fontSize: size * 0.14, fontWeight: 800,
          fontFamily: "var(--font-mono)",
          color: hex, letterSpacing: "-0.03em", lineHeight: 1,
        }}>
          {pct}%
        </div>
        <div style={{
          fontSize: size * 0.045, fontWeight: 700,
          color: hex, letterSpacing: "0.12em",
          textTransform: "uppercase", marginTop: 4, opacity: 0.85,
        }}>
          {label}
        </div>
      </div>
    </div>
  );
}
