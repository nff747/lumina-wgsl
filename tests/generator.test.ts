import { describe, it, expect } from 'vitest';
import { generatePathTracerWGSL } from '../src/path-tracer/generator';

describe('Path Tracer WGSL Generator', () => {
  it('should generate valid WGSL struct strings', () => {
    const wgsl = generatePathTracerWGSL();
    expect(wgsl).toContain('struct Ray');
    expect(wgsl).toContain('struct Camera');
    expect(wgsl).toContain('struct Material');
  });
});

describe('Path Tracer Logic', () => {
  it('should include BRDF and scattering', () => {
    const wgsl = generatePathTracerWGSL();
    expect(wgsl).toContain('fn lambertian_scatter');
  });
});

describe('Path Tracer Entry Point', () => {
  it('should include compute shader main function', () => {
    const wgsl = generatePathTracerWGSL();
    expect(wgsl).toContain('@compute @workgroup_size(16, 16)');
    expect(wgsl).toContain('fn main(');
  });
});
