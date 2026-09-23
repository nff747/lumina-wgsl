export function getTracerWGSL(): string {
  return `
    fn ray_color(ray: Ray, seed: ptr<function, u32>) -> vec3<f32> {
      var current_ray = ray;
      var accumulated_color = vec3<f32>(1.0, 1.0, 1.0);
      
      for (var i = 0; i < 5; i++) {
        // Dummy hit logic
        if (current_ray.direction.y < -0.5) {
           return vec3<f32>(0.0, 0.0, 0.0);
        }
        
        let t = 0.5 * (current_ray.direction.y + 1.0);
        return accumulated_color * mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.5, 0.7, 1.0), t);
      }
      return vec3<f32>(0.0);
    }
  `;
}
