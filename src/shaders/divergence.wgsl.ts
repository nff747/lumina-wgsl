/**
 * Lumina WGSL - Velocity Field Divergence & Boundary Clamping Compute Shader
 * Computes central difference divergence: ∇ · u = (∂u/∂x + ∂v/∂y + ∂w/∂z)
 */
export const divergenceShader = /* wgsl */ `
struct GridParams {
  gridSize: vec3<u32>,
  halfrdx: f32, // 0.5 / dx
};

@group(0) @binding(0) var<uniform> params: GridParams;
@group(0) @binding(1) var velocityTex: texture_3d<f32>;
@group(0) @binding(2) var divergenceTex: texture_storage_3d<r16float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= params.gridSize.x || global_id.y >= params.gridSize.y || global_id.z >= params.gridSize.z) {
    return;
  }

  let coord = vec3<i32>(global_id);
  let maxCoord = vec3<i32>(params.gridSize) - vec3<i32>(1);

  // Neighbor coordinates with boundary clamping
  let left   = vec3<i32>(max(0, coord.x - 1), coord.y, coord.z);
  let right  = vec3<i32>(min(maxCoord.x, coord.x + 1), coord.y, coord.z);
  let bottom = vec3<i32>(coord.x, max(0, coord.y - 1), coord.z);
  let top    = vec3<i32>(coord.x, min(maxCoord.y, coord.y + 1), coord.z);
  let back   = vec3<i32>(coord.x, coord.y, max(0, coord.z - 1));
  let front  = vec3<i32>(coord.x, coord.y, min(maxCoord.z, coord.z + 1));

  let vL = textureLoad(velocityTex, left, 0).x;
  let vR = textureLoad(velocityTex, right, 0).x;
  let vB = textureLoad(velocityTex, bottom, 0).y;
  let vT = textureLoad(velocityTex, top, 0).y;
  let vBack = textureLoad(velocityTex, back, 0).z;
  let vFront = textureLoad(velocityTex, front, 0).z;

  // Central difference divergence
  let div = params.halfrdx * ((vR - vL) + (vT - vB) + (vFront - vBack));

  textureStore(divergenceTex, coord, vec4<f32>(div, 0.0, 0.0, 0.0));
}
`;
