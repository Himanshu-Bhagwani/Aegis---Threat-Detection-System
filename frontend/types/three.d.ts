/* Ambient module shim for three@0.169 (no bundled .d.ts) */
declare module "three" {
  export const BackSide: number;
  export const DoubleSide: number;
  export const FrontSide: number;

  export class Vector3 {
    x: number; y: number; z: number;
    constructor(x?: number, y?: number, z?: number);
    set(x: number, y: number, z: number): this;
    copy(v: Vector3): this;
    add(v: Vector3): this;
    clone(): Vector3;
    normalize(): this;
    multiplyScalar(s: number): this;
    setScalar(s: number): this;
  }

  export class Color {
    constructor(color?: number | string);
    setHex(hex: number): this;
    r: number; g: number; b: number;
  }

  export class Euler {
    x: number; y: number; z: number;
    copy(e: Euler): this;
  }

  export class BufferAttribute {
    constructor(array: Float32Array | Uint16Array, itemSize: number);
    needsUpdate: boolean;
    array: Float32Array | Uint16Array;
  }

  export class Float32BufferAttribute extends BufferAttribute {
    constructor(array: Float32Array | number[], itemSize: number);
  }

  export class BufferGeometry {
    attributes: Record<string, BufferAttribute>;
    setAttribute(name: string, attr: BufferAttribute): this;
    setFromPoints(points: Vector3[]): this;
    dispose(): void;
  }

  export class SphereGeometry  extends BufferGeometry { constructor(r?: number, ws?: number, hs?: number); }
  export class TorusGeometry   extends BufferGeometry { constructor(r?: number, tube?: number, rs?: number, ts?: number, arc?: number); }
  export class RingGeometry    extends BufferGeometry { constructor(inner?: number, outer?: number, seg?: number); }
  export class PlaneGeometry   extends BufferGeometry { constructor(w?: number, h?: number); }
  export class BoxGeometry     extends BufferGeometry { constructor(w?: number, h?: number, d?: number); }
  export class CylinderGeometry extends BufferGeometry { constructor(rt?: number, rb?: number, h?: number, rs?: number); }

  export class Material { transparent: boolean; opacity: number; dispose(): void; }

  export class MeshBasicMaterial extends Material {
    constructor(params?: { color?: number | string; wireframe?: boolean; transparent?: boolean; opacity?: number; side?: number });
    color: Color;
  }
  export class MeshPhongMaterial extends Material {
    constructor(params?: { color?: number | string; emissive?: number; specular?: number; shininess?: number; transparent?: boolean; opacity?: number; side?: number });
    color: Color; emissive: Color; specular: Color;
  }
  export class MeshStandardMaterial extends Material {
    constructor(params?: { color?: number | string; roughness?: number; metalness?: number; transparent?: boolean; opacity?: number });
    color: Color;
  }
  export class PointsMaterial extends Material {
    constructor(params?: { color?: number | string; size?: number; transparent?: boolean; opacity?: number; sizeAttenuation?: boolean });
    color: Color;
  }
  export class LineBasicMaterial extends Material {
    constructor(params?: { color?: number | string; transparent?: boolean; opacity?: number; linewidth?: number });
    color: Color;
  }

  export class Object3D {
    position: Vector3;
    rotation: Euler;
    scale: Vector3;
    children: Object3D[];
    add(...objects: Object3D[]): this;
    remove(...objects: Object3D[]): this;
    lookAt(x: number | Vector3, y?: number, z?: number): void;
    copy(src: Object3D): this;
  }

  export class Mesh<G extends BufferGeometry = BufferGeometry, M extends Material = Material> extends Object3D {
    constructor(geometry?: G, material?: M);
    geometry: G;
    material: M;
  }

  export class Points extends Object3D {
    constructor(geometry?: BufferGeometry, material?: PointsMaterial);
    geometry: BufferGeometry;
    material: PointsMaterial;
  }

  export class Line extends Object3D {
    constructor(geometry?: BufferGeometry, material?: LineBasicMaterial);
    geometry: BufferGeometry;
    material: LineBasicMaterial;
  }

  export class LineSegments extends Object3D {
    constructor(geometry?: BufferGeometry, material?: LineBasicMaterial);
    geometry: BufferGeometry;
    material: LineBasicMaterial;
  }

  export class Light extends Object3D {
    intensity: number;
  }
  export class AmbientLight     extends Light { constructor(color?: number, intensity?: number); }
  export class DirectionalLight extends Light { constructor(color?: number, intensity?: number); }
  export class PointLight       extends Light { constructor(color?: number, intensity?: number, distance?: number); }

  export class Camera extends Object3D {}
  export class PerspectiveCamera extends Camera {
    constructor(fov?: number, aspect?: number, near?: number, far?: number);
    aspect: number;
    fov: number;
    updateProjectionMatrix(): void;
  }

  export class Scene extends Object3D {
    background: Color | null;
  }

  export interface WebGLRendererParameters {
    antialias?: boolean;
    alpha?: boolean;
    canvas?: HTMLCanvasElement;
  }
  export class WebGLRenderer {
    constructor(params?: WebGLRendererParameters);
    domElement: HTMLCanvasElement;
    setSize(w: number, h: number): void;
    setPixelRatio(r: number): void;
    setClearColor(color: number, alpha?: number): void;
    render(scene: Scene, camera: Camera): void;
    dispose(): void;
  }
}
