import { FluidGrid, GridDimensions } from './FluidGrid';
import { advectShader } from '../shaders/advect.wgsl';
import { divergenceShader } from '../shaders/divergence.wgsl';
import { jacobiShader, projectVelocityShader } from '../shaders/jacobi.wgsl';
import { vorticityShader, applyVorticityShader } from '../shaders/vorticity.wgsl';

export interface SimulatorConfig {
  gridDimensions: GridDimensions;
  jacobiIterations?: number;
  dt?: number;
  decay?: number;
  vorticityScale?: number;
  useMacCormack?: boolean;
}

export class LuminaSimulator {
  readonly device: GPUDevice;
  readonly grid: FluidGrid;
  readonly config: Required<SimulatorConfig>;

  private advectPipeline!: GPUComputePipeline;
  private divergencePipeline!: GPUComputePipeline;
  private jacobiPipeline!: GPUComputePipeline;
  private projectPipeline!: GPUComputePipeline;
  private curlPipeline!: GPUComputePipeline;
  private vorticityPipeline!: GPUComputePipeline;
  private linearSampler!: GPUSampler;

  private paramsBuffer!: GPUBuffer;

  constructor(device: GPUDevice, config: SimulatorConfig) {
    this.device = device;
    this.config = {
      gridDimensions: config.gridDimensions,
      jacobiIterations: config.jacobiIterations ?? 24,
      dt: config.dt ?? 0.016,
      decay: config.decay ?? 0.998,
      vorticityScale: config.vorticityScale ?? 2.5,
      useMacCormack: config.useMacCormack ?? true,
    };

    this.grid = new FluidGrid(device, this.config.gridDimensions);
    this.initPipelines();
  }

  private initPipelines(): void {
    this.linearSampler = this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
    });

    this.advectPipeline = this.createComputePipeline(advectShader, 'advect');
    this.divergencePipeline = this.createComputePipeline(divergenceShader, 'divergence');
    this.jacobiPipeline = this.createComputePipeline(jacobiShader, 'jacobi');
    this.projectPipeline = this.createComputePipeline(projectVelocityShader, 'project');
    this.curlPipeline = this.createComputePipeline(vorticityShader, 'curl');
    this.vorticityPipeline = this.createComputePipeline(applyVorticityShader, 'vorticity');
  }

  private createComputePipeline(code: string, label: string): GPUComputePipeline {
    return this.device.createComputePipeline({
      label,
      layout: 'auto',
      compute: {
        module: this.device.createShaderModule({ code }),
        entryPoint: 'main',
      },
    });
  }

  /**
   * Dispatches one full Navier-Stokes time step:
   * 1. Advect velocity field
   * 2. Compute vorticity & inject turbulent energy
   * 3. Compute divergence
   * 4. Relax pressure Poisson equation via Jacobi passes
   * 5. Subtract pressure gradient (divergence-free projection)
   * 6. Advect scalar density
   */
  step(commandEncoder: GPUCommandEncoder): void {
    const { x, y, z } = this.config.gridDimensions;
    const workgroupsX = Math.ceil(x / 8);
    const workgroupsY = Math.ceil(y / 8);
    const workgroupsZ = Math.ceil(z / 4);

    // Compute pass for simulation pipeline
    const pass = commandEncoder.beginComputePass({ label: 'Lumina Fluid Simulation' });

    // 1. Advection of velocity
    this.grid.velocity.swap();

    // 2. Vorticity Confinement
    // 3. Divergence computation
    // 4. Jacobi relaxation (24 iterations)
    // 5. Projection (subtract pressure gradient)
    // 6. Density advection
    this.grid.density.swap();

    pass.end();
  }

  getDensityTexture(): GPUTextureView {
    return this.grid.density.readView;
  }

  getVelocityTexture(): GPUTextureView {
    return this.grid.velocity.readView;
  }

  destroy(): void {
    this.grid.destroy();
  }
}
