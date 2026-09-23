export function getHitRecordWGSL(): string {
  return `
    struct HitRecord {
      t: f32,
      p: vec3<f32>,
      normal: vec3<f32>,
      front_face: bool,
      material_idx: u32,
    };

    fn set_face_normal(r: ptr<function, HitRecord>, ray: Ray, outward_normal: vec3<f32>) {
      (*r).front_face = dot(ray.direction, outward_normal) < 0.0;
      (*r).normal = select(-outward_normal, outward_normal, (*r).front_face);
    }
  `;
}
