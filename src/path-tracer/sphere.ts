export function getSphereWGSL(): string {
  return `
    struct Sphere {
      center: vec3<f32>,
      radius: f32,
      material_idx: u32,
    };

    fn intersect_sphere(ray: Ray, sphere: Sphere, t_min: f32, t_max: f32, t: ptr<function, f32>) -> bool {
      let oc = ray.origin - sphere.center;
      let a = dot(ray.direction, ray.direction);
      let half_b = dot(oc, ray.direction);
      let c = dot(oc, oc) - sphere.radius * sphere.radius;
      let discriminant = half_b * half_b - a * c;

      if (discriminant < 0.0) {
        return false;
      }
      
      let sqrtd = sqrt(discriminant);
      var root = (-half_b - sqrtd) / a;
      if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root) {
          return false;
        }
      }

      *t = root;
      return true;
    }
  `;
}
