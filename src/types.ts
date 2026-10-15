export type RGB = readonly [number, number, number];

export interface FluidConfig {
  simResolution: number;
  dyeResolution: number;
  densityDissipation: number;
  velocityDissipation: number;
  pressure: number;
  pressureIterations: number;
  curl: number;
  splatRadius: number;
  splatForce: number;
  shading: boolean;
  colorful: boolean;
  paused: boolean;
  bloom: boolean;
  bloomIntensity: number;
  bloomThreshold: number;
  sunrays: boolean;
  sunraysWeight: number;
  background: string;
  transparent: boolean;
}

export interface Splat {
  x: number;
  y: number;
  dx: number;
  dy: number;
  color: RGB;
}

export interface FluidEngine {
  update(dt: number, splats: readonly Splat[]): void;
  render(): void;
  resize(): void;
  configure(key: keyof FluidConfig): void;
  capture(): Promise<void>;
  destroy(): void;
}

export type ErrorHandler = (error: unknown) => void;
