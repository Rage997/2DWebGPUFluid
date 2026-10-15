import { DEFAULTS, createFluid } from './engine.js';
import { setupUI, type UIHandle } from './ui.js';
import type { FluidConfig, FluidEngine, RGB, Splat } from './types.js';

function requireCanvas(selector: string): HTMLCanvasElement {
  const el = document.querySelector(selector);
  if (!(el instanceof HTMLCanvasElement)) throw new Error(`Missing required canvas: ${selector}`);
  return el;
}

function requireElement(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing required element: #${id}`);
  return el;
}

const canvas = requireCanvas('#fluid');
const status = requireElement('status');

const config: FluidConfig = { ...DEFAULTS };
if (matchMedia('(pointer: coarse)').matches) config.dyeResolution = 512;

let engine: FluidEngine | null = null;
let ui: UIHandle | null = null;
let frame = 0;
let stopped = false;

interface PointerState {
  x: number;
  y: number;
  color: RGB;
  elapsed: number;
}

const pointers = new Map<number, PointerState>();
const splats: Splat[] = [];

function showError(error: unknown) {
  stopped = true;
  cancelAnimationFrame(frame);
  status.hidden = false;
  status.textContent = error instanceof Error ? error.message : String(error);
  status.setAttribute('role', 'alert');
  console.error(error);
}

function color(): RGB {
  const h = Math.random() * 6;
  const x = 1 - Math.abs(h % 2 - 1);
  const rgb: RGB = h < 1 ? [1, x, 0] : h < 2 ? [x, 1, 0] : h < 3 ? [0, 1, x] : h < 4 ? [0, x, 1] : h < 5 ? [x, 0, 1] : [1, 0, x];
  const [r, g, b] = rgb;
  return [r * 0.15, g * 0.15, b * 0.15];
}

function randomSplats() {
  for (let i = 0, n = 5 + Math.floor(Math.random() * 20); i < n; i++) {
    const [r, g, b] = color();
    splats.push({ x: Math.random(), y: Math.random(), dx: (Math.random() - 0.5) * 1000, dy: (Math.random() - 0.5) * 1000, color: [r * 10, g * 10, b * 10] });
  }
}

function position(event: PointerEvent) {
  const rect = canvas.getBoundingClientRect();
  return { x: (event.clientX - rect.left) / rect.width, y: 1 - (event.clientY - rect.top) / rect.height };
}

canvas.addEventListener('pointerdown', event => {
  if (event.pointerType === 'mouse' && event.button !== 0) return;
  event.preventDefault();
  canvas.focus({ preventScroll: true });
  canvas.setPointerCapture(event.pointerId);
  const pos = position(event);
  pointers.set(event.pointerId, { ...pos, color: color(), elapsed: 0 });
});
canvas.addEventListener('pointermove', event => {
  const pointer = pointers.get(event.pointerId);
  if (!pointer) return;
  const next = position(event);
  const aspect = canvas.clientWidth / canvas.clientHeight;
  const dx = (next.x - pointer.x) * Math.min(aspect, 1) * config.splatForce;
  const dy = (next.y - pointer.y) / Math.max(aspect, 1) * config.splatForce;
  if (dx || dy) splats.push({ ...next, dx, dy, color: pointer.color });
  Object.assign(pointer, next);
});
for (const name of ['pointerup', 'pointercancel', 'lostpointercapture'] as const) {
  canvas.addEventListener(name, event => pointers.delete(event.pointerId));
}
window.addEventListener('blur', () => pointers.clear());
window.addEventListener('keydown', event => {
  if (event.target instanceof HTMLElement && (event.target.closest('input,select,textarea,button') || event.target.isContentEditable)) return;
  if (event.repeat) return;
  if (event.code === 'KeyP') { config.paused = !config.paused; ui?.sync(); }
  if (event.code === 'Space') { event.preventDefault(); randomSplats(); }
});
window.addEventListener('resize', () => {
  if (engine && !stopped) {
    try { engine.resize(); } catch (error) { showError(error); }
  }
});

try {
  const created = await createFluid(canvas, config, showError);
  engine = created;
  if (!stopped) {
    ui = setupUI(config, {
      change(key) { try { created.configure(key); } catch (error) { showError(error); } },
      random: randomSplats,
      async capture() { try { await created.capture(); } catch (error) { showError(error); } },
    });
    status.hidden = true;
    randomSplats();
    let previous = performance.now();
    function tick(now: number) {
      if (stopped) return;
      const dt = Math.min((now - previous) / 1000, 1 / 60);
      previous = now;
      try {
        for (const pointer of pointers.values()) {
          pointer.elapsed += dt;
          if (config.colorful && pointer.elapsed >= 0.1) { pointer.color = color(); pointer.elapsed = 0; }
        }
        created.update(dt, splats);
        splats.length = 0;
        created.render();
        frame = requestAnimationFrame(tick);
      } catch (error) { showError(error); }
    }
    frame = requestAnimationFrame(tick);
  }
} catch (error) { showError(error); }

window.addEventListener('pagehide', event => {
  if (event.persisted) return;
  stopped = true;
  cancelAnimationFrame(frame);
  ui?.destroy();
  engine?.destroy();
});
