export function getRayWGSL(): string {
  return `
    struct Ray {
      origin: vec3<f32>,
      direction: vec3<f32>,
    };

    fn get_ray(origin: vec3<f32>, direction: vec3<f32>) -> Ray {
      var ray: Ray;
      ray.origin = origin;
      ray.direction = direction;
      return ray;
    }
  `;
}
