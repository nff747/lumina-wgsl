/**
 * Lumina WGSL - 3D Double-Buffered (Ping-Pong) Fluid Grid
 * Manages VRAM allocations for 3D textures: velocity, pressure, density, divergence, and curl.
 */
export interface GridDimensions {
  x: number;
  y: number;
  z: number;
}

export interface PingPongTexture {
  read: GPUTexture;
  write: GPUTexture;
  readView: GPUTextureView;
  writeView: GPUTextureView;
  swap: () => void;
}

export class FluidGrid {
  readonly device: GPUDevice;
  readonly dimensions: GridDimensions;
  
  public velocity: PingPongTexture;
  public density: PingPongTexture;
  public pressure: PingPongTexture;
  public divergence: GPUTexture;
  public divergenceView: GPUTextureView;
  public curl: GPUTexture;
  public curlView: GPUTextureView;

  constructor(device: GPUDevice, dimensions: GridDimensions) {
    this.device = device;
    this.dimensions = dimensions;

    this.velocity = this.createPingPong(
      'rgba16float',
      GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
    );

    this.density = this.createPingPong(
      'rgba16float',
      GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
    );

    this.pressure = this.createPingPong(
      'r16float',
      GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
    );

    this.divergence = this.create3DTexture(
      'r16float',
      GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
    );
    this.divergenceView = this.divergence.createView();

    this.curl = this.create3DTexture(
      'rgba16float',
      GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
    );
    this.curlView = this.curl.createView();
  }

  private create3DTexture(format: GPUTextureFormat, usage: GPUTextureUsageFlags): GPUTexture {
    return this.device.createTexture({
      size: [this.dimensions.x, this.dimensions.y, this.dimensions.z],
      dimension: '3d',
      format,
      usage,
    });
  }

  private createPingPong(format: GPUTextureFormat, usage: GPUTextureUsageFlags): PingPongTexture {
    let t0 = this.create3DTexture(format, usage);
    let t1 = this.create3DTexture(format, usage);
    let v0 = t0.createView();
    let v1 = t1.createView();

    return {
      get read() { return t0; },
      get write() { return t1; },
      get readView() { return v0; },
      get writeView() { return v1; },
      swap: () => {
        const tempT = t0; t0 = t1; t1 = tempT;
        const tempV = v0; v0 = v1; v1 = tempV;
      }
    };
  }

  destroy(): void {
    this.velocity.read.destroy();
    this.velocity.write.destroy();
    this.density.read.destroy();
    this.density.write.destroy();
    this.pressure.read.destroy();
    this.pressure.write.destroy();
    this.divergence.destroy();
    this.curl.destroy();
  }
}
