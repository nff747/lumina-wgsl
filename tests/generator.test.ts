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
