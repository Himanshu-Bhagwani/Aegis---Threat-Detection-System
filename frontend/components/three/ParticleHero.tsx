"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

interface Props { width?: number; height?: number }

export default function ParticleHero({ width, height }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;

    const W = width  ?? window.innerWidth;
    const H = height ?? window.innerHeight;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    el.appendChild(renderer.domElement);

    const scene  = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, W / H, 0.1, 1000);
    camera.position.z = 3;

    /* ── Particles ────────────────────────────────────── */
    const COUNT = 1400;
    const positions = new Float32Array(COUNT * 3);
    const velocities: number[] = [];

    for (let i = 0; i < COUNT; i++) {
      positions[i * 3]     = (Math.random() - 0.5) * 10;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 10;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 10;
      velocities.push(
        (Math.random() - 0.5) * 0.002,
        (Math.random() - 0.5) * 0.002,
        (Math.random() - 0.5) * 0.002,
      );
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    const mat = new THREE.PointsMaterial({
      color: 0x3d7fff, size: 0.025, transparent: true, opacity: 0.7,
    });
    const particles = new THREE.Points(geo, mat);
    scene.add(particles);

    /* ── Lines ────────────────────────────────────────── */
    const lineMat   = new THREE.LineBasicMaterial({ color: 0x3d7fff, transparent: true, opacity: 0.12 });
    const lineGeo   = new THREE.BufferGeometry();
    const lineCount = 60;
    const linePos   = new Float32Array(lineCount * 2 * 3);
    lineGeo.setAttribute("position", new THREE.BufferAttribute(linePos, 3));
    const lines     = new THREE.LineSegments(lineGeo, lineMat);
    scene.add(lines);

    /* ── Mouse ────────────────────────────────────────── */
    let mx = 0, my = 0;
    const onMouse = (e: MouseEvent) => {
      mx = (e.clientX / W - 0.5) * 0.5;
      my = (e.clientY / H - 0.5) * 0.5;
    };
    window.addEventListener("mousemove", onMouse);

    /* ── Animate ──────────────────────────────────────── */
    let animId = 0;
    const animate = () => {
      animId = requestAnimationFrame(animate);

      // drift particles
      for (let i = 0; i < COUNT; i++) {
        positions[i * 3]     += velocities[i * 3];
        positions[i * 3 + 1] += velocities[i * 3 + 1];
        positions[i * 3 + 2] += velocities[i * 3 + 2];
        // wrap
        for (let d = 0; d < 3; d++) {
          if (positions[i * 3 + d] > 5)  positions[i * 3 + d] = -5;
          if (positions[i * 3 + d] < -5) positions[i * 3 + d] =  5;
        }
      }
      (geo.attributes.position as THREE.BufferAttribute).needsUpdate = true;

      // update connecting lines between nearby particles
      let li = 0;
      for (let i = 0; i < COUNT && li < lineCount; i++) {
        for (let j = i + 1; j < COUNT && li < lineCount; j++) {
          const dx = positions[i*3]   - positions[j*3];
          const dy = positions[i*3+1] - positions[j*3+1];
          const dz = positions[i*3+2] - positions[j*3+2];
          if (dx*dx + dy*dy + dz*dz < 0.8) {
            linePos[li*6]   = positions[i*3];   linePos[li*6+1] = positions[i*3+1]; linePos[li*6+2] = positions[i*3+2];
            linePos[li*6+3] = positions[j*3];   linePos[li*6+4] = positions[j*3+1]; linePos[li*6+5] = positions[j*3+2];
            li++;
          }
        }
      }
      (lines.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;

      // mouse-reactive rotation
      particles.rotation.y += (mx - particles.rotation.y) * 0.03;
      particles.rotation.x += (-my - particles.rotation.x) * 0.03;
      lines.rotation.copy(particles.rotation);

      renderer.render(scene, camera);
    };
    animate();

    /* ── Resize ───────────────────────────────────────── */
    const onResize = () => {
      const nW = el.offsetWidth || window.innerWidth;
      const nH = el.offsetHeight || window.innerHeight;
      camera.aspect = nW / nH;
      camera.updateProjectionMatrix();
      renderer.setSize(nW, nH);
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("mousemove", onMouse);
      window.removeEventListener("resize", onResize);
      renderer.dispose();
      if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement);
    };
  }, [width, height]);

  return <div ref={mountRef} style={{ width: "100%", height: "100%", position: "absolute", inset: 0 }} />;
}
