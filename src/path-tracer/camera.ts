export function getCameraWGSL(): string {
  return `
    struct Camera {
      origin: vec3<f32>,
      lower_left_corner: vec3<f32>,
      horizontal: vec3<f32>,
      vertical: vec3<f32>,
      u: vec3<f32>,
      v: vec3<f32>,
      w: vec3<f32>,
      lens_radius: f32,
    };
  `;
}
