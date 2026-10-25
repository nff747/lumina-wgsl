import { volumetricLightShader } from '../shaders/volumetricLight.wgsl';

export interface VolumetricLightingConfig {
  stepCount?: number;
  lightPosition: [number, number, number];
  lightColor: [number, number, number];
  ambientColor?: [number, number, number];
  scatteringG?: number; // Asymmetry parameter
  absorption?: number;
  densityScale?: number;
}

export class VolumetricRenderer {
  readonly device: GPUDevice;
  private pipeline!: GPUComputePipeline;
  private uniformBuffer!: GPUBuffer;
  private sampler!: GPUSampler;

  /**
   * Packs volumetric lighting configuration into a 64-byte aligned Float32Array
   * suitable for WebGPU uniform buffer dispatch.
   */
  public static packLightingUniforms(config: VolumetricLightingConfig): Float32Array {
    // 16 floats = 64 bytes (16-byte aligned)
    const buf = new Float32Array(16);
    buf[0] = config.lightPosition[0];
    buf[1] = config.lightPosition[1];
    buf[2] = config.lightPosition[2];
    buf[3] = config.scatteringG ?? 0.75;

    buf[4] = config.lightColor[0];
    buf[5] = config.lightColor[1];
    buf[6] = config.lightColor[2];
    buf[7] = config.densityScale ?? 1.0;

    const amb = config.ambientColor ?? [0.05, 0.05, 0.1];
    buf[8] = amb[0];
    buf[9] = amb[1];
    buf[10] = amb[2];
    buf[11] = config.absorption ?? 0.1;

    buf[12] = config.stepCount ?? 64;
    buf[13] = 0.0;
    buf[14] = 0.0;
    buf[15] = 0.0;

    return buf;
  }

  constructor(device: GPUDevice) {
    this.device = device;
    this.init();
  }

  private init(): void {
    this.sampler = this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
    });

    this.pipeline = this.device.createComputePipeline({
      label: 'Volumetric Light Scatter Pass',
      layout: 'auto',
      compute: {
        module: this.device.createShaderModule({ code: volumetricLightShader }),
        entryPoint: 'main',
      },
    });

    // 144-byte uniform buffer for lighting parameters
    this.uniformBuffer = this.device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  render(
    commandEncoder: GPUCommandEncoder,
    densityTexture: GPUTextureView,
    outputTarget: GPUTextureView,
    width: number,
    height: number
  ): void {
    const pass = commandEncoder.beginComputePass({ label: 'Volumetric Ray-March Pass' });
    pass.setPipeline(this.pipeline);
    pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
    pass.end();
  }

  destroy(): void {
    this.uniformBuffer.destroy();
  }
}
