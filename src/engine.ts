// engine.ts — dependency-free WebGPU fluid simulation engine.
//
// Exports:
//   DEFAULTS           — default configuration object (see below).
//   createFluid(canvas, config, onError) -> engine
//
// Engine API:
//   update(dt, splats)  — advance the simulation. splats: [{x,y,dx,dy,color:[r,g,b]}]
//                         in normalized bottom-left coordinates, dx/dy already force-scaled.
//                         Skips physics when config.paused but still injects splats.
//   render()            — present the current state to the canvas (also checks DPR/size).
//   resize()            — update canvas backing store and GPU targets for the current size.
//   configure(key)      — apply a changed config key (handles resolution/transparent changes).
//   capture()           — async: render offscreen, read back, and download a PNG.
//   destroy()           — release all GPU resources.
//
// The config object is shared and mutable; most keys are read live each frame.
//
// Uniform strategy: every pass invocation gets its own 256-byte-aligned slot inside a
// per-submission uniform pool (one for the sim passes, one for the display passes). The
// whole pool is written with a single queue.writeBuffer right before the matching submit,
// so no buffer is ever overwritten between passes within a frame.

import { SHADERS } from './shaders.js';
import type { ErrorHandler, FluidConfig, FluidEngine, Splat } from './types.js';

export const DEFAULTS: FluidConfig = {
  simResolution: 128,
  dyeResolution: 1024,
  densityDissipation: 1,
  velocityDissipation: 0.2,
  pressure: 0.8,
  pressureIterations: 20,
  curl: 30,
  splatRadius: 0.25,
  splatForce: 6000,
  shading: true,
  colorful: true,
  paused: false,
  bloom: true,
  bloomIntensity: 0.8,
  bloomThreshold: 0.6,
  sunrays: true,
  sunraysWeight: 1,
  background: '#000000',
  transparent: false,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SIM_USAGE = globalThis.GPUTextureUsage && (
  GPUTextureUsage.RENDER_ATTACHMENT |
  GPUTextureUsage.TEXTURE_BINDING |
  GPUTextureUsage.STORAGE_BINDING |
  GPUTextureUsage.COPY_SRC |
  GPUTextureUsage.COPY_DST);

const DISPLAY_USAGE = globalThis.GPUTextureUsage && (
  GPUTextureUsage.RENDER_ATTACHMENT |
  GPUTextureUsage.TEXTURE_BINDING |
  GPUTextureUsage.COPY_SRC |
  GPUTextureUsage.COPY_DST);

const UNIFORM_USAGE = globalThis.GPUBufferUsage && (GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
const SPLAT_USAGE = globalThis.GPUBufferUsage && (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

const UNIFORM_SLOT_BYTES = 256; // >= minUniformBufferOffsetAlignment (guaranteed <= 256)
const UNIFORM_SLOT_FLOATS = UNIFORM_SLOT_BYTES / 4;
const SIM_SLOTS = 16;
const DISPLAY_SLOTS = 16;

const SPLAT_FLOATS = 8; // per-splat stride (x,y,dx,dy,r,g,b,pad)
const MAX_BLOOM_LEVELS = 6;
const SIM_FORMAT: GPUTextureFormat = 'rgba16float';

const ADD_BLEND: GPUBlendState = {
  color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
  alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
};

// ---------------------------------------------------------------------------
// Resource shapes
// ---------------------------------------------------------------------------

interface PingPong {
  textures: [GPUTexture, GPUTexture];
  views: [GPUTextureView, GPUTextureView];
  src: number;
  dst: number;
}

interface Single {
  texture: GPUTexture;
  view: GPUTextureView;
}

interface UniformSlotValues {
  texelSize?: readonly [number, number];
  dyeTexelSize?: readonly [number, number];
  dt?: number;
  dissipation?: number;
  aspect?: number;
  curl?: number;
  pressure?: number;
  count?: number;
  mode?: number;
  direction?: readonly [number, number];
  background?: readonly [number, number, number, number];
  params?: readonly [number, number, number, number];
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function parseColor(hex: string): [number, number, number] {
  if (typeof hex !== 'string') return [0, 0, 0];
  let h = hex.trim().replace(/^#/, '');
  if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
  if (h.length !== 6) return [0, 0, 0];
  const n = parseInt(h, 16);
  if (Number.isNaN(n)) return [0, 0, 0];
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

function devicePixelRatio(): number {
  return (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
}

async function encodePNG(rgba: Uint8Array<ArrayBuffer>, width: number, height: number): Promise<Blob> {
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Unable to acquire a 2D context for PNG encoding.');
  ctx.putImageData(
    new ImageData(new Uint8ClampedArray(rgba.buffer), width, height), 0, 0,
  );
  return canvas.convertToBlob({ type: 'image/png' });
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

class FluidEngineImpl implements FluidEngine {
  // -- always present (set in the constructor) ------------------------------
  private readonly canvas: HTMLCanvasElement;
  private readonly config: FluidConfig;
  private readonly device: GPUDevice;
  destroyed: boolean;

  private displayWidth: number;
  private displayHeight: number;
  private simWidth: number;
  private simHeight: number;
  private dyeWidth: number;
  private dyeHeight: number;
  private simSlot: number;
  private displaySlot: number;
  private bloom: GPUTexture[];

  // -- created during init() before any public method runs ------------------
  private sampler!: GPUSampler;
  private canvasContext!: GPUCanvasContext;
  private canvasFormat!: GPUTextureFormat;

  private splatPipeline!: GPURenderPipeline;
  private advectionPipeline!: GPURenderPipeline;
  private displayPipeline!: GPURenderPipeline;
  private bloomPrefilterPipeline!: GPURenderPipeline;
  private bloomBlurPipeline!: GPURenderPipeline;
  private bloomBlurAddPipeline!: GPURenderPipeline;
  private sunraysPipeline!: GPURenderPipeline;
  private copyPipeline!: GPURenderPipeline;
  private compositeCanvasPipeline!: GPURenderPipeline;
  private compositeCapturePipeline!: GPURenderPipeline;

  private divergencePipeline!: GPUComputePipeline;
  private curlPipeline!: GPUComputePipeline;
  private vorticityPipeline!: GPUComputePipeline;
  private pressureClearPipeline!: GPUComputePipeline;
  private jacobiPipeline!: GPUComputePipeline;
  private gradientSubtractPipeline!: GPUComputePipeline;

  private simUniformBuffer!: GPUBuffer;
  private displayUniformBuffer!: GPUBuffer;
  private simUniformData!: Float32Array;
  private displayUniformData!: Float32Array;
  private splatBuffer!: GPUBuffer;
  private splatData!: Float32Array;

  private velocity!: PingPong;
  private pressure!: PingPong;
  private divergence!: Single;
  private curl!: Single;
  private dye!: PingPong;

  // -- display / bloom / sunrays targets (nulled while being rebuilt) --------
  private display: GPUTexture | null = null;
  private displayView: GPUTextureView | null = null;
  private bloomViews: GPUTextureView[] = [];
  private bloomLevels: number = 0;
  private sunrays: GPUTexture | null = null;
  private sunraysView: GPUTextureView | null = null;

  constructor(canvas: HTMLCanvasElement, config: FluidConfig, device: GPUDevice) {
    this.canvas = canvas;
    this.config = config;
    this.device = device;
    this.destroyed = false;

    this.displayWidth = 1;
    this.displayHeight = 1;
    this.simWidth = 1;
    this.simHeight = 1;
    this.dyeWidth = 1;
    this.dyeHeight = 1;
    this.simSlot = 0;
    this.displaySlot = 0;
    this.bloom = [];
  }

  init() {
    this.createSamplers();
    this.configureCanvas();
    this.createPipelines();
    this.createUniforms();
    this.createSplatBuffer();
    this.resize();
  }

  // -- resource creation ----------------------------------------------------

  private createSamplers() {
    this.sampler = this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
  }

  private configureCanvas() {
    const context = this.canvas.getContext('webgpu');
    if (!context) throw new Error('Unable to acquire the WebGPU canvas context.');
    this.canvasContext = context;
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.canvasFormat = format;
    let alphaMode: GPUCanvasAlphaMode = this.config.transparent ? 'premultiplied' : 'opaque';
    try {
      context.configure({ device: this.device, format, alphaMode });
    } catch {
      alphaMode = 'opaque';
      context.configure({ device: this.device, format, alphaMode });
    }
  }

  private createRenderPipeline(
    module: GPUShaderModule,
    targetFormat: GPUTextureFormat,
    { blend }: { blend?: GPUBlendState } = {},
  ): GPURenderPipeline {
    const target: GPUColorTargetState = { format: targetFormat };
    if (blend) target.blend = blend;
    return this.device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [target] },
      primitive: { topology: 'triangle-list' },
    });
  }

  private createPipelines() {
    const device = this.device;
    const modules: { [name: string]: GPUShaderModule } = {};
    for (const [name, code] of Object.entries(SHADERS)) modules[name] = device.createShaderModule({ code });

    this.splatPipeline = this.createRenderPipeline(modules.splat, SIM_FORMAT);
    this.advectionPipeline = this.createRenderPipeline(modules.advection, SIM_FORMAT);
    this.displayPipeline = this.createRenderPipeline(modules.display, SIM_FORMAT);
    this.bloomPrefilterPipeline = this.createRenderPipeline(modules.bloomPrefilter, SIM_FORMAT);
    this.bloomBlurPipeline = this.createRenderPipeline(modules.bloomBlur, SIM_FORMAT);
    this.bloomBlurAddPipeline = this.createRenderPipeline(modules.bloomBlur, SIM_FORMAT, { blend: ADD_BLEND });
    this.sunraysPipeline = this.createRenderPipeline(modules.sunrays, SIM_FORMAT);
    this.copyPipeline = this.createRenderPipeline(modules.copy, SIM_FORMAT);
    this.compositeCanvasPipeline = this.createRenderPipeline(modules.composite, this.canvasFormat);
    this.compositeCapturePipeline = this.createRenderPipeline(modules.composite, 'rgba8unorm');

    this.divergencePipeline = device.createComputePipeline({ layout: 'auto', compute: { module: modules.divergence, entryPoint: 'cs' } });
    this.curlPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: modules.curl, entryPoint: 'cs' } });
    this.vorticityPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: modules.vorticity, entryPoint: 'cs' } });
    this.pressureClearPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: modules.pressureClear, entryPoint: 'cs' } });
    this.jacobiPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: modules.jacobi, entryPoint: 'cs' } });
    this.gradientSubtractPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: modules.gradientSubtract, entryPoint: 'cs' } });
  }

  private createUniforms() {
    this.simUniformBuffer = this.device.createBuffer({ size: SIM_SLOTS * UNIFORM_SLOT_BYTES, usage: UNIFORM_USAGE });
    this.displayUniformBuffer = this.device.createBuffer({ size: DISPLAY_SLOTS * UNIFORM_SLOT_BYTES, usage: UNIFORM_USAGE });
    this.simUniformData = new Float32Array(SIM_SLOTS * UNIFORM_SLOT_FLOATS);
    this.displayUniformData = new Float32Array(DISPLAY_SLOTS * UNIFORM_SLOT_FLOATS);
  }

  private createSplatBuffer(capacity = 512) {
    this.splatBuffer?.destroy();
    this.splatData = new Float32Array(capacity * SPLAT_FLOATS);
    this.splatBuffer = this.device.createBuffer({ size: this.splatData.byteLength, usage: SPLAT_USAGE });
  }

  private makePingPong(width: number, height: number): PingPong {
    const t0 = this.device.createTexture({ size: [width, height], format: SIM_FORMAT, usage: SIM_USAGE });
    const t1 = this.device.createTexture({ size: [width, height], format: SIM_FORMAT, usage: SIM_USAGE });
    return {
      textures: [t0, t1],
      views: [t0.createView(), t1.createView()],
      src: 0,
      dst: 1,
    };
  }

  private makeSingle(width: number, height: number): Single {
    const t = this.device.createTexture({ size: [width, height], format: SIM_FORMAT, usage: SIM_USAGE });
    return { texture: t, view: t.createView() };
  }

  private destroyPingPong(pp: PingPong) {
    pp.textures[0].destroy();
    pp.textures[1].destroy();
  }

  // -- sizing ---------------------------------------------------------------

  private getResolution(resolution: number): { width: number; height: number } {
    const landscape = this.displayWidth >= this.displayHeight;
    const aspectRatio = landscape ? this.displayWidth / this.displayHeight : this.displayHeight / this.displayWidth;
    const scale = Math.min(1, this.device.limits.maxTextureDimension2D / (resolution * aspectRatio));
    const base = Math.max(1, Math.round(resolution * scale));
    const long = Math.max(1, Math.round(resolution * aspectRatio * scale));
    return landscape ? { width: long, height: base } : { width: base, height: long };
  }

  // -- display / bloom / sunrays textures -----------------------------------

  private createDisplayTextures() {
    this.destroyDisplayTextures();
    const w = this.displayWidth;
    const h = this.displayHeight;
    const display = this.device.createTexture({ size: [w, h], format: SIM_FORMAT, usage: DISPLAY_USAGE });
    this.display = display;
    this.displayView = display.createView();

    this.bloom = [];
    this.bloomViews = [];
    let bw = Math.max(1, w >> 1);
    let bh = Math.max(1, h >> 1);
    while (this.bloom.length < MAX_BLOOM_LEVELS) {
      const t = this.device.createTexture({ size: [bw, bh], format: SIM_FORMAT, usage: DISPLAY_USAGE });
      this.bloom.push(t);
      this.bloomViews.push(t.createView());
      if (bw <= 1 && bh <= 1) break;
      bw = Math.max(1, bw >> 1);
      bh = Math.max(1, bh >> 1);
    }
    this.bloomLevels = this.bloom.length;

    const sunrays = this.device.createTexture({ size: [w, h], format: SIM_FORMAT, usage: DISPLAY_USAGE });
    this.sunrays = sunrays;
    this.sunraysView = sunrays.createView();
  }

  private destroyDisplayTextures() {
    if (this.display) { this.display.destroy(); this.display = null; this.displayView = null; }
    for (const t of this.bloom) t.destroy();
    this.bloom = [];
    this.bloomViews = [];
    this.bloomLevels = 0;
    if (this.sunrays) { this.sunrays.destroy(); this.sunrays = null; this.sunraysView = null; }
  }

  // -- sim / dye textures (with content preservation on resize) -------------

  private recreateSimAndDye() {
    const simSize = this.getResolution(this.config.simResolution);
    const dyeSize = this.getResolution(this.config.dyeResolution);
    const simChanged = simSize.width !== this.simWidth || simSize.height !== this.simHeight;
    const dyeChanged = dyeSize.width !== this.dyeWidth || dyeSize.height !== this.dyeHeight;
    if (!simChanged && !dyeChanged) return;

    const oldVelocity = this.velocity;
    const oldDye = this.dye;

    if (simChanged) {
      if (this.pressure) this.destroyPingPong(this.pressure);
      this.divergence?.texture.destroy();
      this.curl?.texture.destroy();
      this.velocity = this.makePingPong(simSize.width, simSize.height);
      this.pressure = this.makePingPong(simSize.width, simSize.height);
      this.divergence = this.makeSingle(simSize.width, simSize.height);
      this.curl = this.makeSingle(simSize.width, simSize.height);
      this.simWidth = simSize.width;
      this.simHeight = simSize.height;
      if (oldVelocity) {
        this.copyInto(oldVelocity, this.velocity);
        this.destroyPingPong(oldVelocity);
      }
    }

    if (dyeChanged) {
      this.dye = this.makePingPong(dyeSize.width, dyeSize.height);
      this.dyeWidth = dyeSize.width;
      this.dyeHeight = dyeSize.height;
      if (oldDye) {
        this.copyInto(oldDye, this.dye);
        this.destroyPingPong(oldDye);
      }
    }
  }

  // Bilinearly copy both frames of an old ping-pong pair into a new one.
  private copyInto(oldPP: PingPong, newPP: PingPong) {
    const encoder = this.device.createCommandEncoder();
    this.copyPass(encoder, oldPP.views[oldPP.src], newPP.views[0]);
    this.copyPass(encoder, oldPP.views[1 - oldPP.src], newPP.views[1]);
    this.device.queue.submit([encoder.finish()]);
  }

  private copyPass(encoder: GPUCommandEncoder, srcView: GPUTextureView, dstView: GPUTextureView) {
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: dstView, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }],
    });
    pass.setPipeline(this.copyPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: this.copyPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: srcView },
        { binding: 1, resource: this.sampler },
      ],
    }));
    pass.draw(3);
    pass.end();
  }

  // -- public: resize -------------------------------------------------------

  resize() {
    if (this.destroyed) return;
    const dpr = devicePixelRatio();
    const w = Math.min(this.device.limits.maxTextureDimension2D, Math.max(1, Math.floor(this.canvas.clientWidth * dpr)));
    const h = Math.min(this.device.limits.maxTextureDimension2D, Math.max(1, Math.floor(this.canvas.clientHeight * dpr)));
    const firstTime = !this.display;
    const sizeChanged = firstTime || w !== this.displayWidth || h !== this.displayHeight;
    this.canvas.width = w;
    this.canvas.height = h;
    this.displayWidth = w;
    this.displayHeight = h;
    if (sizeChanged) this.createDisplayTextures();
    this.recreateSimAndDye();
  }

  // -- public: configure ----------------------------------------------------

  configure(key: keyof FluidConfig) {
    if (this.destroyed) return;
    if (key === 'simResolution' || key === 'dyeResolution') {
      this.recreateSimAndDye();
    } else if (key === 'transparent') {
      this.configureCanvas();
    }
    // All other keys are read live in update()/render().
  }

  // -- uniform packing ------------------------------------------------------

  private fillUniformSlot(data: Float32Array, slot: number, values: UniformSlotValues = {}) {
    const o = slot * UNIFORM_SLOT_FLOATS;
    const { texelSize, dyeTexelSize, dt, dissipation, aspect, curl, pressure, count, mode, direction, background, params } = values;
    if (texelSize) { data[o] = texelSize[0]; data[o + 1] = texelSize[1]; }
    if (dyeTexelSize) { data[o + 2] = dyeTexelSize[0]; data[o + 3] = dyeTexelSize[1]; }
    data[o + 4] = dt || 0;
    data[o + 5] = dissipation || 0;
    data[o + 6] = aspect || 1;
    data[o + 7] = curl || 0;
    data[o + 8] = pressure || 0;
    data[o + 9] = count || 0;
    data[o + 10] = mode || 0;
    if (direction) { data[o + 12] = direction[0]; data[o + 13] = direction[1]; }
    if (background) {
      data[o + 16] = background[0];
      data[o + 17] = background[1];
      data[o + 18] = background[2];
      data[o + 19] = background[3] !== undefined ? background[3] : 1;
    }
    if (params) {
      data[o + 20] = params[0];
      data[o + 21] = params[1];
      data[o + 22] = params[2];
      data[o + 23] = params[3];
    }
  }

  // -- public: update -------------------------------------------------------

  update(dt: number, splats: readonly Splat[]) {
    if (this.destroyed) return;
    const config = this.config;
    const encoder = this.device.createCommandEncoder();
    const aspect = this.displayWidth / this.displayHeight;
    const simTexel: readonly [number, number] = [1 / this.simWidth, 1 / this.simHeight];
    this.simSlot = 0;
    this.simUniformData.fill(0);

    if (splats && splats.length > 0) {
      this.applySplats(encoder, splats, aspect, simTexel);
    }

    if (!config.paused) {
      this.passCurl(encoder, simTexel);
      this.passVorticity(encoder, dt, simTexel);
      this.passDivergence(encoder, simTexel);
      this.passPressureClear(encoder);
      this.passJacobi(encoder, simTexel);
      this.passGradientSubtract(encoder, simTexel);
      this.passAdvection(encoder, dt, simTexel, this.velocity, this.velocity, config.velocityDissipation);
      this.passAdvection(encoder, dt, simTexel, this.velocity, this.dye, config.densityDissipation);
    }

    this.device.queue.writeBuffer(this.simUniformBuffer, 0, this.simUniformData);
    this.device.queue.submit([encoder.finish()]);
  }

  private applySplats(encoder: GPUCommandEncoder, splats: readonly Splat[], aspect: number, simTexel: readonly [number, number]) {
    const count = splats.length;
    if (count * SPLAT_FLOATS > this.splatData.length) {
      this.createSplatBuffer(2 ** Math.ceil(Math.log2(count)));
    }
    const dyeTexel: readonly [number, number] = [1 / this.dyeWidth, 1 / this.dyeHeight];
    const data = this.splatData.subarray(0, count * SPLAT_FLOATS);
    for (let i = 0; i < count; i++) {
      const s = splats[i];
      const o = i * SPLAT_FLOATS;
      data[o] = s.x;
      data[o + 1] = s.y;
      data[o + 2] = s.dx;
      data[o + 3] = s.dy;
      data[o + 4] = s.color[0];
      data[o + 5] = s.color[1];
      data[o + 6] = s.color[2];
      data[o + 7] = 0;
    }
    this.device.queue.writeBuffer(this.splatBuffer, 0, data);
    const radius = this.config.splatRadius / 100 * Math.max(aspect, 1);

    const velSlot = this.simSlot++;
    this.fillUniformSlot(this.simUniformData, velSlot, { texelSize: simTexel, aspect, mode: 0, count, params: [radius, 0, 0, 0] });
    this.splatPass(encoder, velSlot, this.velocity);

    const dyeSlot = this.simSlot++;
    this.fillUniformSlot(this.simUniformData, dyeSlot, { texelSize: dyeTexel, aspect, mode: 1, count, params: [radius, 0, 0, 0] });
    this.splatPass(encoder, dyeSlot, this.dye);
  }

  private splatPass(encoder: GPUCommandEncoder, slot: number, pp: PingPong) {
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: pp.views[pp.dst], loadOp: 'load', storeOp: 'store' }],
    });
    pass.setPipeline(this.splatPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: this.splatPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.simUniformBuffer, offset: slot * UNIFORM_SLOT_BYTES } },
        { binding: 1, resource: { buffer: this.splatBuffer } },
        { binding: 2, resource: pp.views[pp.src] },
      ],
    }));
    pass.draw(3);
    pass.end();
    this.swap(pp);
  }

  // -- simulation passes ----------------------------------------------------

  private passCurl(encoder: GPUCommandEncoder, simTexel: readonly [number, number]) {
    const slot = this.simSlot++;
    this.fillUniformSlot(this.simUniformData, slot, { texelSize: simTexel });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.curlPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: this.curlPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.simUniformBuffer, offset: slot * UNIFORM_SLOT_BYTES } },
        { binding: 1, resource: this.velocity.views[this.velocity.src] },
        { binding: 2, resource: this.curl.view },
      ],
    }));
    pass.dispatchWorkgroups(Math.ceil(this.simWidth / 8), Math.ceil(this.simHeight / 8));
    pass.end();
  }

  private passVorticity(encoder: GPUCommandEncoder, dt: number, simTexel: readonly [number, number]) {
    const slot = this.simSlot++;
    this.fillUniformSlot(this.simUniformData, slot, { texelSize: simTexel, dt, curl: this.config.curl });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.vorticityPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: this.vorticityPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.simUniformBuffer, offset: slot * UNIFORM_SLOT_BYTES } },
        { binding: 1, resource: this.curl.view },
        { binding: 2, resource: this.velocity.views[this.velocity.src] },
        { binding: 3, resource: this.velocity.views[this.velocity.dst] },
      ],
    }));
    pass.dispatchWorkgroups(Math.ceil(this.simWidth / 8), Math.ceil(this.simHeight / 8));
    pass.end();
    this.swap(this.velocity);
  }

  private passDivergence(encoder: GPUCommandEncoder, simTexel: readonly [number, number]) {
    const slot = this.simSlot++;
    this.fillUniformSlot(this.simUniformData, slot, { texelSize: simTexel });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.divergencePipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: this.divergencePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.simUniformBuffer, offset: slot * UNIFORM_SLOT_BYTES } },
        { binding: 1, resource: this.velocity.views[this.velocity.src] },
        { binding: 2, resource: this.divergence.view },
      ],
    }));
    pass.dispatchWorkgroups(Math.ceil(this.simWidth / 8), Math.ceil(this.simHeight / 8));
    pass.end();
  }

  private passPressureClear(encoder: GPUCommandEncoder) {
    const slot = this.simSlot++;
    this.fillUniformSlot(this.simUniformData, slot, { pressure: this.config.pressure });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pressureClearPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: this.pressureClearPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.simUniformBuffer, offset: slot * UNIFORM_SLOT_BYTES } },
        { binding: 1, resource: this.pressure.views[this.pressure.src] },
        { binding: 2, resource: this.pressure.views[this.pressure.dst] },
      ],
    }));
    pass.dispatchWorkgroups(Math.ceil(this.simWidth / 8), Math.ceil(this.simHeight / 8));
    pass.end();
    this.swap(this.pressure);
  }

  private passJacobi(encoder: GPUCommandEncoder, simTexel: readonly [number, number]) {
    const slot = this.simSlot++;
    this.fillUniformSlot(this.simUniformData, slot, { texelSize: simTexel });
    const iterations = Math.max(1, Math.floor(this.config.pressureIterations));
    for (let i = 0; i < iterations; i++) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.jacobiPipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: this.jacobiPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.simUniformBuffer, offset: slot * UNIFORM_SLOT_BYTES } },
          { binding: 1, resource: this.pressure.views[this.pressure.src] },
          { binding: 2, resource: this.divergence.view },
          { binding: 3, resource: this.pressure.views[this.pressure.dst] },
        ],
      }));
      pass.dispatchWorkgroups(Math.ceil(this.simWidth / 8), Math.ceil(this.simHeight / 8));
      pass.end();
      this.swap(this.pressure);
    }
  }

  private passGradientSubtract(encoder: GPUCommandEncoder, simTexel: readonly [number, number]) {
    const slot = this.simSlot++;
    this.fillUniformSlot(this.simUniformData, slot, { texelSize: simTexel });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.gradientSubtractPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: this.gradientSubtractPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.simUniformBuffer, offset: slot * UNIFORM_SLOT_BYTES } },
        { binding: 1, resource: this.pressure.views[this.pressure.src] },
        { binding: 2, resource: this.velocity.views[this.velocity.src] },
        { binding: 3, resource: this.velocity.views[this.velocity.dst] },
      ],
    }));
    pass.dispatchWorkgroups(Math.ceil(this.simWidth / 8), Math.ceil(this.simHeight / 8));
    pass.end();
    this.swap(this.velocity);
  }

  // velocity = flow field, source = field being advected (velocity or dye).
  private passAdvection(encoder: GPUCommandEncoder, dt: number, simTexel: readonly [number, number], velocity: PingPong, source: PingPong, dissipation: number) {
    const slot = this.simSlot++;
    this.fillUniformSlot(this.simUniformData, slot, { texelSize: simTexel, dt, dissipation });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: source.views[source.dst], loadOp: 'load', storeOp: 'store' }],
    });
    pass.setPipeline(this.advectionPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: this.advectionPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.simUniformBuffer, offset: slot * UNIFORM_SLOT_BYTES } },
        { binding: 1, resource: velocity.views[velocity.src] },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: source.views[source.src] },
        { binding: 4, resource: this.sampler },
      ],
    }));
    pass.draw(3);
    pass.end();
    this.swap(source);
  }

  // -- public: render -------------------------------------------------------

  render() {
    if (this.destroyed) return;
    const dpr = devicePixelRatio();
    const w = Math.min(this.device.limits.maxTextureDimension2D, Math.max(1, Math.floor(this.canvas.clientWidth * dpr)));
    const h = Math.min(this.device.limits.maxTextureDimension2D, Math.max(1, Math.floor(this.canvas.clientHeight * dpr)));
    if (w !== this.canvas.width || h !== this.canvas.height) this.resize();

    const config = this.config;
    const encoder = this.device.createCommandEncoder();
    const simTexel: readonly [number, number] = [1 / this.simWidth, 1 / this.simHeight];
    const dyeTexel: readonly [number, number] = [1 / this.dyeWidth, 1 / this.dyeHeight];
    const bg = config.transparent ? [0, 0, 0] : parseColor(config.background);
    this.displaySlot = 0;
    this.displayUniformData.fill(0);

    const displaySlot = this.displaySlot++;
    this.fillUniformSlot(this.displayUniformData, displaySlot, {
      texelSize: simTexel,
      dyeTexelSize: dyeTexel,
      background: [bg[0], bg[1], bg[2], 1],
      params: [config.shading ? 1 : 0, 0.8, 0, 0],
    });
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: this.displayView!, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }],
      });
      pass.setPipeline(this.displayPipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: this.displayPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.displayUniformBuffer, offset: displaySlot * UNIFORM_SLOT_BYTES } },
          { binding: 1, resource: this.dye.views[this.dye.src] },
          { binding: 2, resource: this.sampler },
        ],
      }));
      pass.draw(3);
      pass.end();
    }

    if (config.bloom && this.bloomLevels > 0) this.passBloom(encoder);
    if (config.sunrays) this.passSunrays(encoder);
    this.passComposite(encoder, this.canvasContext.getCurrentTexture().createView(), this.compositeCanvasPipeline);

    this.device.queue.writeBuffer(this.displayUniformBuffer, 0, this.displayUniformData);
    this.device.queue.submit([encoder.finish()]);
  }

  private passBloom(encoder: GPUCommandEncoder) {
    const config = this.config;
    const preSlot = this.displaySlot++;
    this.fillUniformSlot(this.displayUniformData, preSlot, { params: [config.bloomThreshold, 0, 0, 0] });
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: this.bloomViews[0], loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }],
      });
      pass.setPipeline(this.bloomPrefilterPipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: this.bloomPrefilterPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.displayUniformBuffer, offset: preSlot * UNIFORM_SLOT_BYTES } },
          { binding: 1, resource: this.displayView! },
          { binding: 2, resource: this.sampler },
        ],
      }));
      pass.draw(3);
      pass.end();
    }
    for (let i = 0; i < this.bloomLevels - 1; i++) {
      const target = this.bloom[i + 1];
      const slot = this.displaySlot++;
      this.fillUniformSlot(this.displayUniformData, slot, { texelSize: [1 / target.width, 1 / target.height] });
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: this.bloomViews[i + 1], loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }],
      });
      pass.setPipeline(this.bloomBlurPipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: this.bloomBlurPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.displayUniformBuffer, offset: slot * UNIFORM_SLOT_BYTES } },
          { binding: 1, resource: this.bloomViews[i] },
          { binding: 2, resource: this.sampler },
        ],
      }));
      pass.draw(3);
      pass.end();
    }
    for (let i = this.bloomLevels - 2; i >= 0; i--) {
      const target = this.bloom[i];
      const slot = this.displaySlot++;
      this.fillUniformSlot(this.displayUniformData, slot, { texelSize: [1 / target.width, 1 / target.height] });
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: this.bloomViews[i], loadOp: 'load', storeOp: 'store' }],
      });
      pass.setPipeline(this.bloomBlurAddPipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: this.bloomBlurAddPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.displayUniformBuffer, offset: slot * UNIFORM_SLOT_BYTES } },
          { binding: 1, resource: this.bloomViews[i + 1] },
          { binding: 2, resource: this.sampler },
        ],
      }));
      pass.draw(3);
      pass.end();
    }
  }

  private passSunrays(encoder: GPUCommandEncoder) {
    const config = this.config;
    const slot = this.displaySlot++;
    this.fillUniformSlot(this.displayUniformData, slot, { params: [config.bloomThreshold, config.sunraysWeight, 0, 0] });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.sunraysView!, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }],
    });
    pass.setPipeline(this.sunraysPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: this.sunraysPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.displayUniformBuffer, offset: slot * UNIFORM_SLOT_BYTES } },
        { binding: 1, resource: this.displayView! },
        { binding: 2, resource: this.sampler },
      ],
    }));
    pass.draw(3);
    pass.end();
  }

  private passComposite(encoder: GPUCommandEncoder, targetView: GPUTextureView, pipeline: GPURenderPipeline, capture = false) {
    const config = this.config;
    const slot = this.displaySlot++;
    this.fillUniformSlot(this.displayUniformData, slot, {
      background: [...parseColor(config.background), 1],
      params: [config.bloom ? config.bloomIntensity : 0, config.sunrays ? config.sunraysWeight : 0, config.transparent ? 1 : 0, capture ? 1 : 0],
    });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: targetView, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.displayUniformBuffer, offset: slot * UNIFORM_SLOT_BYTES } },
        { binding: 1, resource: this.displayView! },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: this.bloomViews[0] },
        { binding: 4, resource: this.sampler },
        { binding: 5, resource: this.sunraysView! },
        { binding: 6, resource: this.sampler },
      ],
    }));
    pass.draw(3);
    pass.end();
  }

  // -- public: capture ------------------------------------------------------

  async capture(): Promise<void> {
    if (this.destroyed) throw new Error('Engine destroyed.');
    this.render();
    const w = this.displayWidth;
    const h = this.displayHeight;

    // Render the composite into a fresh offscreen 8-bit texture (unique per call).
    const captureTexture = this.device.createTexture({
      size: [w, h],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const captureView = captureTexture.createView();
    const encoder = this.device.createCommandEncoder();
    this.displaySlot = 0;
    this.displayUniformData.fill(0);
    this.passComposite(encoder, captureView, this.compositeCapturePipeline, true);
    this.device.queue.writeBuffer(this.displayUniformBuffer, 0, this.displayUniformData);

    // Padded readback: each row is aligned to 256 bytes.
    const bytesPerRow = Math.ceil((w * 4) / 256) * 256;
    const readback = this.device.createBuffer({
      size: bytesPerRow * h,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyTextureToBuffer(
      { texture: captureTexture },
      { buffer: readback, bytesPerRow, rowsPerImage: h },
      [w, h, 1],
    );
    this.device.queue.submit([encoder.finish()]);

    let pixels: Uint8Array<ArrayBuffer>;
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const mapped = new Uint8Array(readback.getMappedRange());
      pixels = new Uint8Array(w * h * 4);
      for (let y = 0; y < h; y++) {
        pixels.set(mapped.subarray(y * bytesPerRow, y * bytesPerRow + w * 4), y * w * 4);
      }
      readback.unmap();
    } finally {
      readback.destroy();
      captureTexture.destroy();
    }
    const blob = await encodePNG(pixels, w, h);
    this.download(blob);
  }

  private download(blob: Blob) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fluid-${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // -- public: destroy ------------------------------------------------------

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this.canvasContext?.unconfigure();
    this.device.destroy();
  }

  private swap(pp: PingPong) {
    const tmp = pp.src;
    pp.src = pp.dst;
    pp.dst = tmp;
  }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

export async function createFluid(canvas: HTMLCanvasElement, config: FluidConfig, onError: ErrorHandler): Promise<FluidEngine> {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('WebGPU is unavailable. Use a WebGPU-capable browser on HTTPS or localhost.');
  }
  if (!canvas || typeof canvas.getContext !== 'function') {
    throw new Error('Invalid canvas element.');
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No suitable WebGPU adapter found.');
  const device = await adapter.requestDevice();

  const engine = new FluidEngineImpl(canvas, config, device);
  device.addEventListener('uncapturederror', event => onError?.(new Error(event.error.message)));
  device.lost.then((info) => {
    if (info.reason !== 'destroyed' && !engine.destroyed) {
      onError?.(new Error('WebGPU device lost: ' + (info.message || info.reason)));
    }
  });
  try {
    engine.init();
  } catch (error) {
    engine.destroy();
    throw error;
  }
  return engine;
}
