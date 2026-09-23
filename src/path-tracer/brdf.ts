export function getBRDFWGSL(): string {
  return `
    fn lambertian_scatter(normal: vec3<f32>, seed: ptr<function, u32>) -> vec3<f32> {
      // random point in unit sphere
      let theta = rand(seed) * 2.0 * 3.14159265;
      let phi = acos(2.0 * rand(seed) - 1.0);
      let r = pow(rand(seed), 0.333333);
      
      let x = r * sin(phi) * cos(theta);
      let y = r * sin(phi) * sin(theta);
      let z = r * cos(phi);
      
      let scatter_dir = normal + vec3<f32>(x, y, z);
      return normalize(scatter_dir);
    }
  `;
}
