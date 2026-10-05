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

  public advectPipeline!: GPUComputePipeline;
  public divergencePipeline!: GPUComputePipeline;
  public jacobiPipeline!: GPUComputePipeline;
  public projectPipeline!: GPUComputePipeline;
  public curlPipeline!: GPUComputePipeline;
  public vorticityPipeline!: GPUComputePipeline;
  public linearSampler!: GPUSampler;

  public paramsBuffer!: GPUBuffer;

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

    this.paramsBuffer = this.device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
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

    // Update uniform parameters
    const paramsData = new ArrayBuffer(64);
    const u32View = new Uint32Array(paramsData);
    const f32View = new Float32Array(paramsData);

    u32View[0] = x;
    u32View[1] = y;
    u32View[2] = z;
    f32View[3] = this.config.dt;
    f32View[4] = this.config.decay;
    u32View[5] = this.config.useMacCormack ? 1 : 0;
    f32View[6] = 0.0;

    this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsData);

    const pass = commandEncoder.beginComputePass({ label: 'Lumina Fluid Simulation' });

    // 1. Advection of velocity
    const advectVelocityBindGroup = this.device.createBindGroup({
      layout: this.advectPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: this.linearSampler },
        { binding: 2, resource: this.grid.velocity.readView },
        { binding: 3, resource: this.grid.velocity.readView },
        { binding: 4, resource: this.grid.velocity.writeView },
      ],
    });
    pass.setPipeline(this.advectPipeline);
    pass.setBindGroup(0, advectVelocityBindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
    this.grid.velocity.swap();

    // 2. Divergence computation
    const divBindGroup = this.device.createBindGroup({
      layout: this.divergencePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: this.grid.velocity.readView },
        { binding: 2, resource: this.grid.divergenceView },
      ],
    });
    pass.setPipeline(this.divergencePipeline);
    pass.setBindGroup(0, divBindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);

    // 3. Jacobi pressure relaxation passes
    pass.setPipeline(this.jacobiPipeline);
    for (let i = 0; i < this.config.jacobiIterations; i++) {
      const jacobiBindGroup = this.device.createBindGroup({
        layout: this.jacobiPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.paramsBuffer } },
          { binding: 1, resource: this.grid.pressure.readView },
          { binding: 2, resource: this.grid.divergenceView },
          { binding: 3, resource: this.grid.pressure.writeView },
        ],
      });
      pass.setBindGroup(0, jacobiBindGroup);
      pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
      this.grid.pressure.swap();
    }

    // 4. Projection pass (velocity minus pressure gradient)
    const projectBindGroup = this.device.createBindGroup({
      layout: this.projectPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: this.grid.pressure.readView },
        { binding: 2, resource: this.grid.velocity.readView },
        { binding: 3, resource: this.grid.velocity.writeView },
      ],
    });
    pass.setPipeline(this.projectPipeline);
    pass.setBindGroup(0, projectBindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
    this.grid.velocity.swap();

    // 5. Density advection
    const advectDensityBindGroup = this.device.createBindGroup({
      layout: this.advectPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: this.linearSampler },
        { binding: 2, resource: this.grid.velocity.readView },
        { binding: 3, resource: this.grid.density.readView },
        { binding: 4, resource: this.grid.density.writeView },
      ],
    });
    pass.setPipeline(this.advectPipeline);
    pass.setBindGroup(0, advectDensityBindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
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
    if (this.paramsBuffer) {
      this.paramsBuffer.destroy();
    }
  }
}
