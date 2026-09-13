/**
 * Lumina WGSL - Thermal Buoyancy Compute Shader (Boussinesq Approximation)
 * Applies upward vertical buoyant lift based on fluid temperature and density:
 * F_buoyancy = (-alpha * density + beta * (temperature - ambientTemperature)) * vec3(0, 1, 0)
 */
export const buoyancyShader = /* wgsl */ `
struct BuoyancyParams {
  gridSize: vec3<u32>,
  ambientTemperature: f32,
  alpha: f32, // Density weight
  beta: f32,  // Temperature thermal expansion coefficient
  dt: f32,
};

@group(0) @binding(0) var<uniform> params: BuoyancyParams;
@group(0) @binding(1) var densityTex: texture_3d<f32>;
@group(0) @binding(2) var temperatureTex: texture_3d<f32>;
@group(0) @binding(3) var velocityTex: texture_3d<f32>;
@group(0) @binding(4) var outputVelocity: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= params.gridSize.x || global_id.y >= params.gridSize.y || global_id.z >= params.gridSize.z) {
    return;
  }

  let coord = vec3<i32>(global_id);
  let density = textureLoad(densityTex, coord, 0).r;
  let temperature = textureLoad(temperatureTex, coord, 0).r;
  var velocity = textureLoad(velocityTex, coord, 0).xyz;

  // Compute thermal buoyancy force
  let forceY = (-params.alpha * density + params.beta * (temperature - params.ambientTemperature));
  velocity.y += forceY * params.dt;

  textureStore(outputVelocity, coord, vec4<f32>(velocity, 0.0));
}
`;
