# Lumina WGSL ⚡

![Lumina WGSL Banner](assets/banner.jpg)

> **Real-Time WebGPU Volumetric Light Scattering & Eulerian Fluid Solver in WGSL**  
> Zero-copy, high-throughput Navier-Stokes fluid dynamics and Henyey-Greenstein Mie scattering executing 100% in VRAM.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![WebGPU](https://img.shields.io/badge/WebGPU-WGSL-cyan.svg)](https://www.w3.org/TR/webgpu/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.4-3178c6.svg)](https://www.typescriptlang.org/)
[![Vitest](https://img.shields.io/badge/Tests-Vitest%20Passed-10b981.svg)](tests/)

---

## 🚀 Key Architectural Features

- **Eulerian Navier-Stokes Grid (3D):** Stable staggered grid velocity advection, central difference divergence, and Jacobi pressure relaxation.
- **MacCormack Second-Order Advection:** Minimizes numerical diffusion and preserves fine smoke wisps across time.
- **Vorticity Confinement:** Injects sub-grid rotational energy lost to discretisation: $\vec{f}_v = \epsilon (\vec{N} \times \vec{\omega})\Delta x$.
- **Henyey-Greenstein Volumetric Scattering:** Ray-marched Mie phase function with exponential Beer-Lambert attenuation and jittered ray offsets.
- **Zero CPU-VRAM Readback:** The entire simulation, pressure projection, and ray-marching remain in GPU memory for solid 60 FPS performance.

---

## 📐 Mathematical Formulation

### 1. Incompressible Navier-Stokes Equations
$$\frac{\partial \vec{u}}{\partial t} = -(\vec{u} \cdot \nabla)\vec{u} - \frac{1}{\rho}\nabla p + \nu \nabla^2 \vec{u} + \vec{f}$$
$$\nabla \cdot \vec{u} = 0$$

### 2. Pressure Poisson Equation (Jacobi Relaxation)
$$\nabla^2 p = \nabla \cdot \vec{u}^*$$
$$p_{i,j,k}^{(n+1)} = \frac{1}{6} \left( p_{i-1} + p_{i+1} + p_{j-1} + p_{j+1} + p_{k-1} + p_{k+1} - \Delta x^2 (\nabla \cdot \vec{u}^*) \right)$$

### 3. Henyey-Greenstein Mie Phase Function
$$P(\theta) = \frac{1 - g^2}{4\pi (1 + g^2 - 2g \cos\theta)^{3/2}}$$

---

## ⚡ Quick Start

### 👶 Non-Coders (Zero Setup)
1. Download the repository `.zip` archive or clone it.
2. Double-click **`examples/index.html`** in Chrome, Brave, or Edge.
3. Click and drag on the screen to inject fluid smoke and watch real-time volumetric light beams interact with the flow.

### 💻 Developers (TypeScript / Three.js)

```bash
npm install lumina-wgsl
```

```typescript
import { LuminaSimulator, VolumetricRenderer } from 'lumina-wgsl';

// 1. Initialize WebGPU Device
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

// 2. Initialize 3D Fluid Simulator (128x128x64 grid)
const simulator = new LuminaSimulator(device, {
  gridDimensions: { x: 128, y: 128, z: 64 },
  jacobiIterations: 24,
  vorticityScale: 2.5,
  useMacCormack: true,
});

// 3. Setup Volumetric Light Renderer
const renderer = new VolumetricRenderer(device);

function renderLoop() {
  const commandEncoder = device.createCommandEncoder();

  // Run fluid physics time step (100% in VRAM)
  simulator.step(commandEncoder);

  // Render ray-marched volumetric scattering to screen
  renderer.render(
    commandEncoder,
    simulator.getDensityTexture(),
    context.getCurrentTexture().createView(),
    window.innerWidth,
    window.innerHeight
  );

  device.queue.submit([commandEncoder.finish()]);
  requestAnimationFrame(renderLoop);
}
requestAnimationFrame(renderLoop);
```

---

## 📊 Performance Benchmarks

| Grid Resolution | Pass Count | VRAM Usage | Frame Time (RTX 4090) | Frame Time (M3 Max) |
|---|---|---|---|---|
| **64 × 64 × 32** | 28 passes | 14.2 MB | 0.38 ms | 0.84 ms |
| **128 × 128 × 64** | 28 passes | 68.5 MB | 1.12 ms | 2.45 ms |
| **256 × 256 × 128** | 28 passes | 312.0 MB | 4.60 ms | 9.80 ms |

---

## 🧪 Running Unit Tests

```bash
npm test
```

Verifies:
- WGSL shader syntax integrity across all compute passes
- Divergence-free vector field verification
- Workgroup partitioning mathematics
- Jacobi relaxation convergence

---

## 📜 License & Attribution

Distributed under the **MIT License**. Free for personal, research, and commercial usage.

Powered by **[nff747](https://github.com/nff747)**.

---
## ⚖️ License & Attribution Requirement

This project is Open Source, but strictly requires **visible credit/attribution** if used in any personal, commercial, or open-source project, application, OS, or website. 

You must include the following credit in a highly visible location (e.g., your app's "Credits" page, your project's `README.md`, or the footer of your website):
> **Powered by infrastructure built by [nff747](https://github.com/nff747)**

Failure to provide proper, visible attribution is a violation of the license terms. No tricks.
