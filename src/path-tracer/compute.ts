export function getComputeWGSL(): string {
  return `
    @group(0) @binding(0) var output_tex: texture_storage_2d<rgba8unorm, write>;

    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let dims = textureDimensions(output_tex);
      if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
      }
      
      let pixel_coords = vec2<i32>(global_id.xy);
      
      var seed = global_id.x + global_id.y * dims.x;
      // initialize ray
      
      textureStore(output_tex, pixel_coords, vec4<f32>(1.0, 0.0, 0.0, 1.0));
    }
  `;
}
