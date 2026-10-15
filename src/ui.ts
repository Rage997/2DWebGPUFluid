import type { FluidConfig } from './types.js';

const SMALL_SCREEN = '(max-width: 640px)';

const CONFIG_KEYS = [
  'simResolution',
  'dyeResolution',
  'densityDissipation',
  'velocityDissipation',
  'pressure',
  'pressureIterations',
  'curl',
  'splatRadius',
  'splatForce',
  'shading',
  'colorful',
  'paused',
  'bloom',
  'bloomIntensity',
  'bloomThreshold',
  'sunrays',
  'sunraysWeight',
  'background',
  'transparent',
] as const satisfies readonly (keyof FluidConfig)[];

function isConfigKey(value: string | undefined): value is keyof FluidConfig {
  return value !== undefined && (CONFIG_KEYS as readonly string[]).includes(value);
}

function formatValue(value: number | boolean | string): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return String(value);
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

function requireElement(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing required element: #${id}`);
  return el;
}

function requireButton(id: string): HTMLButtonElement {
  const el = document.getElementById(id);
  if (!(el instanceof HTMLButtonElement)) throw new Error(`Missing required element: #${id}`);
  return el;
}

type Control = HTMLInputElement | HTMLSelectElement;

interface Field {
  el: Control;
  output: HTMLOutputElement | null;
}

export interface UIActions {
  change(key: keyof FluidConfig): void;
  random(): void;
  capture(): Promise<void>;
}

export interface UIHandle {
  sync(): void;
  destroy(): void;
}

export function setupUI(config: FluidConfig, actions: UIActions): UIHandle {
  const panel = requireElement('panel');
  const body = requireElement('panel-body');
  const toggle = requireElement('panel-toggle');
  const randomBtn = requireButton('btn-random');
  const captureBtn = requireButton('btn-capture');

  const fields = new Map<keyof FluidConfig, Field>();
  for (const el of body.querySelectorAll('[data-key]')) {
    const control = el instanceof HTMLInputElement || el instanceof HTMLSelectElement ? el : null;
    const key = control ? control.dataset.key : undefined;
    if (!control || !isConfigKey(key)) continue;
    const row = control.closest('.row');
    fields.set(key, {
      el: control,
      output: row ? row.querySelector('output') : null,
    });
  }

  let collapsed = matchMedia(SMALL_SCREEN).matches;
  let capturing = false;

  function readConfigValue(key: keyof FluidConfig, el: Control): void {
    switch (key) {
      case 'background':
        config[key] = el.value;
        break;
      case 'shading':
      case 'colorful':
      case 'paused':
      case 'bloom':
      case 'sunrays':
      case 'transparent':
        if (el instanceof HTMLInputElement) config[key] = el.checked;
        break;
      default:
        config[key] = Number(el.value);
    }
  }

  function updateDisabled() {
    const background = fields.get('background');
    if (background) background.el.disabled = !!config.transparent;
    const bloom = fields.get('bloom');
    if (bloom) {
      const on = !!config.bloom;
      const intensity = fields.get('bloomIntensity');
      const threshold = fields.get('bloomThreshold');
      if (intensity) intensity.el.disabled = !on;
      if (threshold) threshold.el.disabled = !on;
    }
    const sunrays = fields.get('sunrays');
    if (sunrays) {
      const weight = fields.get('sunraysWeight');
      if (weight) weight.el.disabled = !config.sunrays;
    }
  }

  function applyCollapsed() {
    panel.classList.toggle('is-collapsed', collapsed);
    toggle.setAttribute('aria-expanded', String(!collapsed));
    toggle.textContent = collapsed ? 'Show' : 'Hide';
  }

  function sync() {
    for (const [key, field] of fields) {
      const value = config[key];
      const { el, output } = field;
      if (el instanceof HTMLInputElement && el.type === 'checkbox') el.checked = !!value;
      else if (el.value !== String(value)) el.value = String(value);
      if (output) output.textContent = formatValue(value);
    }
    updateDisabled();
    applyCollapsed();
  }

  function onInput(event: Event) {
    const target = event.target;
    const el = target instanceof HTMLInputElement || target instanceof HTMLSelectElement ? target : null;
    const key = el ? el.dataset.key : undefined;
    if (!el || !isConfigKey(key)) return;
    readConfigValue(key, el);
    const field = fields.get(key);
    if (field?.output) field.output.textContent = formatValue(config[key]);
    updateDisabled();
    actions.change(key);
  }

  function onToggle() {
    collapsed = !collapsed;
    applyCollapsed();
  }
  const onRandom = () => actions.random();

  async function onCapture() {
    if (capturing) return;
    capturing = true;
    captureBtn.disabled = true;
    try {
      await actions.capture();
    } finally {
      capturing = false;
      captureBtn.disabled = false;
    }
  }

  function onKeyDown(event: KeyboardEvent) {
    if (event.repeat) return;
    const target = event.target;
    if (target instanceof HTMLElement &&
        (target.closest('input,select,textarea,button') || target.isContentEditable)) return;
    if (event.code === 'KeyH') onToggle();
  }

  body.addEventListener('input', onInput);
  toggle.addEventListener('click', onToggle);
  randomBtn.addEventListener('click', onRandom);
  captureBtn.addEventListener('click', onCapture);
  window.addEventListener('keydown', onKeyDown);

  sync();

  return {
    sync,
    destroy() {
      body.removeEventListener('input', onInput);
      toggle.removeEventListener('click', onToggle);
      randomBtn.removeEventListener('click', onRandom);
      captureBtn.removeEventListener('click', onCapture);
      window.removeEventListener('keydown', onKeyDown);
    },
  };
}
