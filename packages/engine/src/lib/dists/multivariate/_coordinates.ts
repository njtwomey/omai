import { sliceAxis, squeeze, type Value } from '../../tensor/index.ts'

/** Coordinate i of points stored on the last axis. */
export const coordinate = (u: Value, i: number): Value => squeeze(sliceAxis(u, -1, i, i + 1), -1)
