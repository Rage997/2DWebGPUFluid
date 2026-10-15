# 2D WebGPU Fluid

Native WebGPU reimplementation of [Pavel Dobryakov's WebGL Fluid Simulation](https://github.com/PavelDoGreat/WebGL-Fluid-Simulation).

## Run

Requires **Node.js 24 or newer**

```sh
npm ci
npm run dev
```

Open the localhost URL printed by Vite in a WebGPU-capable browser

```sh
npm run typecheck  # Strict TypeScript checking without emitting files
npm run build      # Type check, then build static assets into dist/
npm run preview    # Serve the production build locally
```

## Controls

- Drag with the mouse or one or more fingers to inject color and momentum.
- **P** toggles simulation pause. Input still adds splats while paused.
- **Space** injects random splats.
- **H**, or the panel's Show/Hide button, toggles controls. The panel starts collapsed on narrow screens.
- Keyboard shortcuts do not intercept input while a form control has focus.
- **Save PNG** downloads the canvas image without the controls, at the current canvas backing resolution. Enable **Transparent background** for an RGBA PNG.
- Separate simulation/dye resolutions let you trade flow detail and visual detail against GPU cost. The values specify the shorter texture dimension; the longer dimension follows the canvas aspect ratio, bounded by GPU limits. Resolution changes resample existing dye and velocity.
- Density/velocity dissipation fade dye and damp motion; these are not physical diffusion/viscosity coefficients. Pressure controls retention of the prior pressure estimate; pressure iterations control the approximate solve. Curl reinforces vortices.

## Implementation

| File | Responsibility |
| --- | --- |
| `src/app.ts` | Startup, pointer capture and multitouch, keyboard shortcuts, frame timing |
| `src/engine.ts` | Device and resources, pipeline scheduling, ping-pong fields, resizing, GPU readback and PNG capture |
| `src/shaders.ts` | WGSL compute and full-screen rendering shaders |
| `src/ui.ts` | Typed native control bindings and panel state |
| `src/types.ts` | Shared configuration, RGB tuple, splat, error-handler, and engine contracts |
| `index.html`, `styles.css` | Accessible controls and responsive layout |
| `vite.config.ts`, `tsconfig.json` | Build target and strict browser TypeScript configuration |

The engine exposes `createFluid(canvas, config, onError)` and the `FluidEngine` interface. Configuration is shared and mutable; call `configure(key)` after changes that affect GPU resources. `destroy()` releases the device. GPU implementation details stay private. Type-only imports do not add runtime code.

`@webgpu/types` supplies WebGPU constants and canvas overloads missing from TypeScript's DOM declarations. `skipLibCheck` avoids checking overlapping dependency declarations; all application source remains strictly checked.

The solver is a two-dimensional incompressible-flow approximation: splats → curl → vorticity confinement → divergence → pressure attenuation → Jacobi pressure iterations → gradient subtraction → velocity advection → dye advection. There is no explicit viscosity solve or 3D liquid surface.

Simulation fields use `rgba16float` textures. Stencil operations run in 8×8 compute workgroups; advection and splats use render passes. Field updates ping-pong between textures. Each pass has its own aligned uniform slot so later CPU writes cannot overwrite another pass's parameters within a submission. Pointer splats are uploaded together in a reusable, growable storage buffer.

## Attribution

The numerical scheme is adapted from the MIT-licensed original by Pavel Dobryakov. Its copyright and license are retained in `LICENSE`. The original also references [GPU Gems, Chapter 38: Fast Fluid Dynamics Simulation on the GPU](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu).
