export function getMaterialWGSL(): string {
  return `
    struct Material {
      base_color: vec3<f32>,
      roughness: f32,
      metallic: f32,
      emission: vec3<f32>,
      transmission: f32,
      ior: f32,
    };
  `;
}
