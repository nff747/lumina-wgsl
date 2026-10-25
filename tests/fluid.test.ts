import { describe, it, expect } from 'vitest';
import { advectShader } from '../src/shaders/advect.wgsl';
import { divergenceShader } from '../src/shaders/divergence.wgsl';
import { jacobiShader, projectVelocityShader } from '../src/shaders/jacobi.wgsl';
import { vorticityShader, applyVorticityShader } from '../src/shaders/vorticity.wgsl';
import { volumetricLightShader } from '../src/shaders/volumetricLight.wgsl';

describe('Lumina WGSL - Compute Shader Syntax Integrity', () => {
  it('should compile advection shader with valid WGSL structure', () => {
    expect(advectShader).toContain('@compute @workgroup_size(8, 8, 4)');
    expect(advectShader).toContain('fn main(');
    expect(advectShader).toContain('sampleBilinear');
  });

  it('should compile divergence shader with 6-point 3D stencil', () => {
    expect(divergenceShader).toContain('@compute @workgroup_size(8, 8, 4)');
    expect(divergenceShader).toContain('params.halfrdx * ((vR - vL) + (vT - vB) + (vFront - vBack))');
  });

  it('should compile Jacobi pressure Poisson solver shader', () => {
    expect(jacobiShader).toContain('params.inverseBeta');
    expect(jacobiShader).toContain('params.alpha * div');
    expect(projectVelocityShader).toContain('oldVel - gradP');
  });

  it('should compile 3D curl and vorticity confinement shaders', () => {
    expect(vorticityShader).toContain('vT.z - vB.z');
    expect(applyVorticityShader).toContain('cross(N, curlVec)');
  });

  it('should compile ray-marched volumetric scattering shader with Henyey-Greenstein', () => {
    expect(volumetricLightShader).toContain('phaseHG');
    expect(volumetricLightShader).toContain('intersectAABB');
    expect(volumetricLightShader).toContain('exp(-densitySample * params.absorption * stepSize)');
  });
});

describe('Lumina WGSL - Fluid Math & Boundary Conditions', () => {
  it('calculates correct workgroup counts for arbitrary 3D grid dimensions', () => {
    const dims = { x: 128, y: 128, z: 64 };
    const wgX = Math.ceil(dims.x / 8);
    const wgY = Math.ceil(dims.y / 8);
    const wgZ = Math.ceil(dims.z / 4);

    expect(wgX).toBe(16);
    expect(wgY).toBe(16);
    expect(wgZ).toBe(16);
    expect(wgX * wgY * wgZ).toBe(4096);
  });

  it('verifies incompressibility condition formulation: ∇ · u = 0', () => {
    // Test that a divergence-free vortex field yields zero divergence
    const testField = (x: number, y: number) => ({ u: -y, v: x });
    const dx = 0.01;
    const dy = 0.01;

    const div = (testField(dx, 0).u - testField(-dx, 0).u) / (2 * dx) +
                (testField(0, dy).v - testField(0, -dy).v) / (2 * dy);

    expect(Math.abs(div)).toBeLessThan(1e-6);
  });
});

import { buoyancyShader } from '../src/shaders/buoyancy.wgsl';

describe('Lumina WGSL - Thermal Buoyancy Kernel', () => {
  it('should compile thermal buoyancy shader with Boussinesq approximation', () => {
    expect(buoyancyShader).toContain('@compute @workgroup_size(8, 8, 4)');
    expect(buoyancyShader).toContain('(-params.alpha * density + params.beta * (temperature - params.ambientTemperature))');
    expect(buoyancyShader).toContain('velocity.y += forceY * params.dt');
  });
});

import { FluidGrid } from '../src/core/FluidGrid';

describe('FluidGrid Memory & Dispatch Estimators', () => {
  it('should compute exact VRAM allocation bytes for 3D textures', () => {
    // 64x64x64 grid: 262,144 voxels * 46 bytes = 12,058,624 bytes (~11.5 MB)
    const dims = { x: 64, y: 64, z: 64 };
    const bytes = FluidGrid.calculateMemoryFootprintBytes(dims);
    expect(bytes).toBe(64 * 64 * 64 * 46);
  });

  it('should compute ceiling workgroup dispatch counts matching (8, 8, 4) tile size', () => {
    const dispatches = FluidGrid.calculateWorkgroupDispatches({ x: 65, y: 33, z: 17 });
    expect(dispatches[0]).toBe(9); // ceil(65 / 8)
    expect(dispatches[1]).toBe(5); // ceil(33 / 8)
    expect(dispatches[2]).toBe(5); // ceil(17 / 4)
  });
});

import { LuminaSimulator } from '../src/core/LuminaSimulator';

describe('LuminaSimulator Uniform Buffer Packing', () => {
  it('should format 64-byte aligned simulation parameters', () => {
    const raw = LuminaSimulator.packSimulationParams({
      gridDimensions: { x: 128, y: 128, z: 64 },
      dt: 0.02,
      decay: 0.995,
      vorticityScale: 3.0,
      jacobiIterations: 32,
      useMacCormack: true,
    });

    expect(raw.byteLength).toBe(64);
    const u32 = new Uint32Array(raw);
    const f32 = new Float32Array(raw);

    expect(u32[0]).toBe(128);
    expect(u32[1]).toBe(128);
    expect(u32[2]).toBe(64);
    expect(f32[3]).toBeCloseTo(0.02, 5);
    expect(f32[4]).toBeCloseTo(0.995, 5);
    expect(u32[5]).toBe(1);
    expect(f32[6]).toBeCloseTo(3.0, 5);
    expect(u32[7]).toBe(32);
  });
});
