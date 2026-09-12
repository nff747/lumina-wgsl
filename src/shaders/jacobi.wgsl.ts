/**
 * Lumina WGSL - Iterative Jacobi Pressure Poisson Equation Solver
 * Solves: ∇²p = ∇ · u on 3D staggered grid
 */
export const jacobiShader = /* wgsl */ `
struct JacobiParams {
  gridSize: vec3<u32>,
  alpha: f32, // -(dx * dx)
  inverseBeta: f32, // 1.0 / 6.0 for 3D 6-point stencil
};

@group(0) @binding(0) var<uniform> params: JacobiParams;
@group(0) @binding(1) var pressureTex: texture_3d<f32>;
@group(0) @binding(2) var divergenceTex: texture_3d<f32>;
@group(0) @binding(3) var outputPressure: texture_storage_3d<r16float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= params.gridSize.x || global_id.y >= params.gridSize.y || global_id.z >= params.gridSize.z) {
    return;
  }

  let coord = vec3<i32>(global_id);
  let maxCoord = vec3<i32>(params.gridSize) - vec3<i32>(1);

  // 6-point 3D stencil with Neumann boundary conditions (dp/dn = 0)
  let pL     = textureLoad(pressureTex, vec3<i32>(max(0, coord.x - 1), coord.y, coord.z), 0).x;
  let pR     = textureLoad(pressureTex, vec3<i32>(min(maxCoord.x, coord.x + 1), coord.y, coord.z), 0).x;
  let pB     = textureLoad(pressureTex, vec3<i32>(coord.x, max(0, coord.y - 1), coord.z), 0).x;
  let pT     = textureLoad(pressureTex, vec3<i32>(coord.x, min(maxCoord.y, coord.y + 1), coord.z), 0).x;
  let pBack  = textureLoad(pressureTex, vec3<i32>(coord.x, coord.y, max(0, coord.z - 1)), 0).x;
  let pFront = textureLoad(pressureTex, vec3<i32>(coord.x, coord.y, min(maxCoord.z, coord.z + 1)), 0).x;

  let div = textureLoad(divergenceTex, coord, 0).x;

  // Jacobi iteration step: p_new = (sum_neighbors + alpha * divergence) * beta
  let pNew = (pL + pR + pB + pT + pBack + pFront + params.alpha * div) * params.inverseBeta;

  textureStore(outputPressure, coord, vec4<f32>(pNew, 0.0, 0.0, 0.0));
}
`;

/**
 * Pressure Gradient Subtraction compute shader
 * Enforces incompressibility: u_new = u - ∇p
 */
export const projectVelocityShader = /* wgsl */ `
struct ProjectParams {
  gridSize: vec3<u32>,
  halfrdx: f32, // 0.5 / dx
};

@group(0) @binding(0) var<uniform> params: ProjectParams;
@group(0) @binding(1) var pressureTex: texture_3d<f32>;
@group(0) @binding(2) var velocityTex: texture_3d<f32>;
@group(0) @binding(3) var outputVelocity: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= params.gridSize.x || global_id.y >= params.gridSize.y || global_id.z >= params.gridSize.z) {
    return;
  }

  let coord = vec3<i32>(global_id);
  let maxCoord = vec3<i32>(params.gridSize) - vec3<i32>(1);

  let pL     = textureLoad(pressureTex, vec3<i32>(max(0, coord.x - 1), coord.y, coord.z), 0).x;
  let pR     = textureLoad(pressureTex, vec3<i32>(min(maxCoord.x, coord.x + 1), coord.y, coord.z), 0).x;
  let pB     = textureLoad(pressureTex, vec3<i32>(coord.x, max(0, coord.y - 1), coord.z), 0).x;
  let pT     = textureLoad(pressureTex, vec3<i32>(coord.x, min(maxCoord.y, coord.y + 1), coord.z), 0).x;
  let pBack  = textureLoad(pressureTex, vec3<i32>(coord.x, coord.y, max(0, coord.z - 1)), 0).x;
  let pFront = textureLoad(pressureTex, vec3<i32>(coord.x, coord.y, min(maxCoord.z, coord.z + 1)), 0).x;

  let gradP = vec3<f32>(
    (pR - pL) * params.halfrdx,
    (pT - pB) * params.halfrdx,
    (pFront - pBack) * params.halfrdx
  );

  let oldVel = textureLoad(velocityTex, coord, 0).xyz;
  var newVel = oldVel - gradP;

  // Enforce zero normal velocity at boundaries
  if (coord.x == 0 && newVel.x < 0.0) { newVel.x = 0.0; }
  if (coord.x == maxCoord.x && newVel.x > 0.0) { newVel.x = 0.0; }
  if (coord.y == 0 && newVel.y < 0.0) { newVel.y = 0.0; }
  if (coord.y == maxCoord.y && newVel.y > 0.0) { newVel.y = 0.0; }
  if (coord.z == 0 && newVel.z < 0.0) { newVel.z = 0.0; }
  if (coord.z == maxCoord.z && newVel.z > 0.0) { newVel.z = 0.0; }

  textureStore(outputVelocity, coord, vec4<f32>(newVel, 0.0));
}
`;
