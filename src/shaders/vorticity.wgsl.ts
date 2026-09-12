/**
 * Lumina WGSL - Vorticity Confinement & Turbulence Recovery Shader
 * Restores sub-grid rotational momentum lost to numerical dissipation.
 * Curl: ω = ∇ × u
 * Force: f_vorticity = ε (N × ω) dx, where N = ∇|ω| / |∇|ω||
 */
export const vorticityShader = /* wgsl */ `
struct VorticityParams {
  gridSize: vec3<u32>,
  halfrdx: f32,
  confinementScale: f32,
  dt: f32,
};

@group(0) @binding(0) var<uniform> params: VorticityParams;
@group(0) @binding(1) var velocityTex: texture_3d<f32>;
@group(0) @binding(2) var curlTex: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 8, 4)
fn computeCurl(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= params.gridSize.x || global_id.y >= params.gridSize.y || global_id.z >= params.gridSize.z) {
    return;
  }

  let coord = vec3<i32>(global_id);
  let maxCoord = vec3<i32>(params.gridSize) - vec3<i32>(1);

  let vL     = textureLoad(velocityTex, vec3<i32>(max(0, coord.x - 1), coord.y, coord.z), 0).xyz;
  let vR     = textureLoad(velocityTex, vec3<i32>(min(maxCoord.x, coord.x + 1), coord.y, coord.z), 0).xyz;
  let vB     = textureLoad(velocityTex, vec3<i32>(coord.x, max(0, coord.y - 1), coord.z), 0).xyz;
  let vT     = textureLoad(velocityTex, vec3<i32>(coord.x, min(maxCoord.y, coord.y + 1), coord.z), 0).xyz;
  let vBack  = textureLoad(velocityTex, vec3<i32>(coord.x, coord.y, max(0, coord.z - 1)), 0).xyz;
  let vFront = textureLoad(velocityTex, vec3<i32>(coord.x, coord.y, min(maxCoord.z, coord.z + 1)), 0).xyz;

  // Curl = (∂w/∂y - ∂v/∂z, ∂u/∂z - ∂w/∂x, ∂v/∂x - ∂u/∂y)
  let curl = vec3<f32>(
    ((vT.z - vB.z) - (vFront.y - vBack.y)) * params.halfrdx,
    ((vFront.x - vBack.x) - (vR.z - vL.z)) * params.halfrdx,
    ((vR.y - vL.y) - (vT.x - vB.x)) * params.halfrdx
  );

  let curlMag = length(curl);
  textureStore(curlTex, coord, vec4<f32>(curl, curlMag));
}
`;

export const applyVorticityShader = /* wgsl */ `
struct VorticityParams {
  gridSize: vec3<u32>,
  halfrdx: f32,
  confinementScale: f32,
  dt: f32,
};

@group(0) @binding(0) var<uniform> params: VorticityParams;
@group(0) @binding(1) var curlTex: texture_3d<f32>;
@group(0) @binding(2) var velocityTex: texture_3d<f32>;
@group(0) @binding(3) var outputVelocity: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 8, 4)
fn applyForce(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= params.gridSize.x || global_id.y >= params.gridSize.y || global_id.z >= params.gridSize.z) {
    return;
  }

  let coord = vec3<i32>(global_id);
  let maxCoord = vec3<i32>(params.gridSize) - vec3<i32>(1);

  let cL     = textureLoad(curlTex, vec3<i32>(max(0, coord.x - 1), coord.y, coord.z), 0).w;
  let cR     = textureLoad(curlTex, vec3<i32>(min(maxCoord.x, coord.x + 1), coord.y, coord.z), 0).w;
  let cB     = textureLoad(curlTex, vec3<i32>(coord.x, max(0, coord.y - 1), coord.z), 0).w;
  let cT     = textureLoad(curlTex, vec3<i32>(coord.x, min(maxCoord.y, coord.y + 1), coord.z), 0).w;
  let cBack  = textureLoad(curlTex, vec3<i32>(coord.x, coord.y, max(0, coord.z - 1)), 0).w;
  let cFront = textureLoad(curlTex, vec3<i32>(coord.x, coord.y, min(maxCoord.z, coord.z + 1)), 0).w;

  // Gradient of curl magnitude
  var eta = vec3<f32>(
    (cR - cL) * params.halfrdx,
    (cT - cB) * params.halfrdx,
    (cFront - cBack) * params.halfrdx
  );

  let etaLength = length(eta);
  let N = select(vec3<f32>(0.0), eta / etaLength, etaLength > 0.00001);

  let curlVec = textureLoad(curlTex, coord, 0).xyz;
  let force = cross(N, curlVec) * params.confinementScale;

  let currentVel = textureLoad(velocityTex, coord, 0).xyz;
  let newVel = currentVel + force * params.dt;

  textureStore(outputVelocity, coord, vec4<f32>(newVel, 0.0));
}
`;
