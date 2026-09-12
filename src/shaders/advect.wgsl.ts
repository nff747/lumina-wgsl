/**
 * Lumina WGSL - Semi-Lagrangian and MacCormack Advection Compute Shader
 * Handles stable velocity and scalar density transport with boundary clamping.
 */
export const advectShader = /* wgsl */ `
struct SimulationParams {
  gridSize: vec3<u32>,
  dt: f32,
  decay: f32,
  macCormack: u32,
  forwardTime: f32,
};

@group(0) @binding(0) var<uniform> params: SimulationParams;
@group(0) @binding(1) var velocitySampler: sampler;
@group(0) @binding(2) var inputVelocity: texture_3d<f32>;
@group(0) @binding(3) var inputQuantity: texture_3d<f32>;
@group(0) @binding(4) var outputQuantity: texture_storage_3d<rgba16float, write>;

fn sampleBilinear(tex: texture_3d<f32>, uvw: vec3<f32>) -> vec4<f32> {
  let clampedUVW = clamp(uvw, vec3<f32>(0.5) / vec3<f32>(params.gridSize), vec3<f32>(1.0) - vec3<f32>(0.5) / vec3<f32>(params.gridSize));
  return textureSampleLevel(tex, velocitySampler, clampedUVW, 0.0);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= params.gridSize.x || global_id.y >= params.gridSize.y || global_id.z >= params.gridSize.z) {
    return;
  }

  let dim = vec3<f32>(params.gridSize);
  let cellPos = (vec3<f32>(global_id) + 0.5) / dim;

  // Read local velocity
  let vel = textureLoad(inputVelocity, vec3<i32>(global_id), 0).xyz;

  // Backtrace coordinates (Semi-Lagrangian step)
  let prevPos = cellPos - (vel * params.dt) / dim;

  var advectedVal = sampleBilinear(inputQuantity, prevPos);

  // Optional MacCormack second-order correction step
  if (params.macCormack == 1u) {
    let forwardVel = sampleBilinear(inputVelocity, prevPos).xyz;
    let reprojectedPos = prevPos + (forwardVel * params.dt) / dim;
    let errorEstimate = (sampleBilinear(inputQuantity, reprojectedPos) - textureLoad(inputQuantity, vec3<i32>(global_id), 0)) * 0.5;
    advectedVal = advectedVal - errorEstimate;
  }

  // Apply fluid dissipation/decay
  let finalVal = advectedVal * params.decay;

  textureStore(outputQuantity, vec3<i32>(global_id), finalVal);
}
`;
