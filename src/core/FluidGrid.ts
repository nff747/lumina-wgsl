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

  /**
   * Calculates the total VRAM footprint in bytes for all 3D fluid grid simulation textures.
   */
  public static calculateMemoryFootprintBytes(dimensions: GridDimensions): number {
    const voxels = dimensions.x * dimensions.y * dimensions.z;
    // rgba16float: 4 channels * 2 bytes = 8 bytes per voxel
    // r16float: 1 channel * 2 bytes = 2 bytes per voxel
    // PingPong velocity: 2 textures * 8 bytes
    // PingPong density: 2 textures * 8 bytes
    // PingPong pressure: 2 textures * 2 bytes
    // Divergence: 1 texture * 2 bytes
    // Curl: 1 texture * 8 bytes
    const totalBytesPerVoxel = (2 * 8) + (2 * 8) + (2 * 2) + 2 + 8; // 16 + 16 + 4 + 2 + 8 = 46 bytes
    return voxels * totalBytesPerVoxel;
  }

  /**
   * Calculates 3D compute workgroup dispatch dimensions based on (8, 8, 4) workgroup tile size.
   */
  public static calculateWorkgroupDispatches(dimensions: GridDimensions): [number, number, number] {
    return [
      Math.ceil(dimensions.x / 8),
      Math.ceil(dimensions.y / 8),
      Math.ceil(dimensions.z / 4)
    ];
  }

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
