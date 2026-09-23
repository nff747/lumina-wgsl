import { getRayWGSL } from './ray';
import { getCameraWGSL } from './camera';
import { getRandomWGSL } from './random';
import { getMaterialWGSL } from './material';
import { getSphereWGSL } from './sphere';
import { getHitRecordWGSL } from './hit';
import { getBRDFWGSL } from './brdf';
import { getTracerWGSL } from './tracer';
import { getComputeWGSL } from './compute';

export function generatePathTracerWGSL(): string {
  return [
    getRandomWGSL(),
    getRayWGSL(),
    getCameraWGSL(),
    getMaterialWGSL(),
    getHitRecordWGSL(),
    getSphereWGSL(),
    getBRDFWGSL(),
    getTracerWGSL(),
    getComputeWGSL(),
  ].join('\n');
}
