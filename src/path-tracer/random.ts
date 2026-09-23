export function getRandomWGSL(): string {
  return `
    fn pcg_hash(seed: ptr<function, u32>) -> u32 {
      *seed = *seed * 747796405u + 2891336453u;
      var word: u32 = ((*seed >> ((*seed >> 28u) + 4u)) ^ *seed) * 277803737u;
      return (word >> 22u) ^ word;
    }

    fn rand(seed: ptr<function, u32>) -> f32 {
      return f32(pcg_hash(seed)) / 4294967295.0;
    }
  `;
}
