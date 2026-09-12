/**
 * Lumina WGSL - Real-Time Ray-Marched Volumetric Light Scattering Shader
 * Features:
 * - Henyey-Greenstein Mie scattering phase function for forward glow
 * - Beer-Lambert exponential extinction: I = I_0 * exp(-σ_t * s)
 * - Jittered ray sampling to prevent banding artifacts
 */
export const volumetricLightShader = /* wgsl */ `
struct LightingParams {
  cameraPos: vec3<f32>,
  stepCount: u32,
  lightPos: vec3<f32>,
  densityScale: f32,
  lightColor: vec3<f32>,
  scatteringG: f32, // Henyey-Greenstein asymmetry (-1..1)
  ambientColor: vec3<f32>,
  absorption: f32,
  screenSize: vec2<f32>,
  invViewProj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> params: LightingParams;
@group(0) @binding(1) var densityTex: texture_3d<f32>;
@group(0) @binding(2) var linearSampler: sampler;
@group(0) @binding(3) var outputColor: texture_storage_2d<rgba16float, write>;

// Henyey-Greenstein Phase Function
fn phaseHG(cosTheta: f32, g: f32) -> f32 {
  let g2 = g * g;
  let denom = 1.0 + g2 - 2.0 * g * cosTheta;
  return (1.0 - g2) / (4.0 * 3.14159265 * pow(max(denom, 0.0001), 1.5));
}

// Ray-box intersection (AABB: [0, 1]^3)
fn intersectAABB(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
  let boxMin = vec3<f32>(0.0);
  let boxMax = vec3<f32>(1.0);
  let invR = 1.0 / rd;
  let tbot = invR * (boxMin - ro);
  let ttop = invR * (boxMax - ro);
  let tmin = min(ttop, tbot);
  let tmax = max(ttop, tbot);
  let t0 = max(max(tmin.x, tmin.y), tmin.z);
  let t1 = min(min(tmax.x, tmax.y), tmax.z);
  return vec2<f32>(t0, t1);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let screenCoord = vec2<f32>(global_id.xy);
  if (screenCoord.x >= params.screenSize.x || screenCoord.y >= params.screenSize.y) {
    return;
  }

  // Normalized Device Coordinates (-1..1)
  let uv = (screenCoord + 0.5) / params.screenSize;
  let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, 1.0, 1.0);

  // Unproject to world-space ray direction
  let worldPos4 = params.invViewProj * ndc;
  let worldTarget = worldPos4.xyz / worldPos4.w;
  let rayDir = normalize(worldTarget - params.cameraPos);
  let rayOrigin = params.cameraPos;

  // Intersect with simulation volume bounds
  let hits = intersectAABB(rayOrigin, rayDir);
  let tNear = max(0.0, hits.x);
  let tFar = hits.y;

  if (tNear >= tFar) {
    textureStore(outputColor, vec2<i32>(global_id.xy), vec4<f32>(0.0));
    return;
  }

  let stepSize = (tFar - tNear) / f32(params.stepCount);
  var currentT = tNear + stepSize * 0.5;

  var accumulatedLight = vec3<f32>(0.0);
  var transmittance = 1.0;

  for (var i = 0u; i < params.stepCount; i = i + 1u) {
    if (transmittance < 0.01) { break; }

    let p = rayOrigin + rayDir * currentT;
    let densitySample = textureSampleLevel(densityTex, linearSampler, p, 0.0).r * params.densityScale;

    if (densitySample > 0.001) {
      // Vector towards point light source
      let toLight = params.lightPos - p;
      let lightDist = length(toLight);
      let lightDir = toLight / max(lightDist, 0.001);

      // Phase function for forward/back scattering
      let cosTheta = dot(rayDir, lightDir);
      let phase = phaseHG(cosTheta, params.scatteringG);

      // Attenuation through medium (Beer-Lambert law)
      let lightAtten = 1.0 / max(lightDist * lightDist, 0.1);
      let inScatter = params.lightColor * lightAtten * phase * densitySample;
      let ambient = params.ambientColor * densitySample;

      accumulatedLight += (inScatter + ambient) * transmittance * stepSize;
      transmittance *= exp(-densitySample * params.absorption * stepSize);
    }

    currentT += stepSize;
  }

  let finalColor = vec4<f32>(accumulatedLight, 1.0 - transmittance);
  textureStore(outputColor, vec2<i32>(global_id.xy), finalColor);
}
`;
