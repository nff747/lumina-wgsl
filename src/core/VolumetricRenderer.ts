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
