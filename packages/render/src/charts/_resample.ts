// Internal: resampling of gridded values for display (corner-aligned, so grid nodes keep their values).
import type { Grid2D } from './_data.ts'

/** How a grid is resampled for display. */
export type Resampling = 'nearest' | 'bilinear' | 'bicubic'

function cubic(p0: number, p1: number, p2: number, p3: number, t: number): number {
  // Catmull-Rom: passes through p1 at t = 0 and p2 at t = 1.
  return p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)))
}

/**
 * Resample a grid to `rows` by `cols` with corners aligned to the input's corner nodes. NaN propagates to the output
 * pixels it touches. Bicubic output is clamped to the input's range so ramps never overshoot.
 */
export function resample(g: Grid2D, rows: number, cols: number, method: Resampling): Float64Array {
  const out = new Float64Array(rows * cols)
  const at = (r: number, c: number): number => {
    const rr = Math.min(g.rows - 1, Math.max(0, r))
    const cc = Math.min(g.cols - 1, Math.max(0, c))
    return g.values[rr * g.cols + cc] ?? Number.NaN
  }
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const v of g.values)
    if (Number.isFinite(v)) {
      lo = Math.min(lo, v)
      hi = Math.max(hi, v)
    }
  for (let i = 0; i < rows; i++) {
    const u = rows === 1 ? 0 : (i * (g.rows - 1)) / (rows - 1)
    const r0 = Math.floor(u)
    const fr = u - r0
    for (let j = 0; j < cols; j++) {
      const w = cols === 1 ? 0 : (j * (g.cols - 1)) / (cols - 1)
      const c0 = Math.floor(w)
      const fc = w - c0
      let v: number
      if (method === 'nearest') v = at(Math.round(u), Math.round(w))
      else if (method === 'bilinear') {
        const a = at(r0, c0) + (at(r0, c0 + 1) - at(r0, c0)) * fc
        const b = at(r0 + 1, c0) + (at(r0 + 1, c0 + 1) - at(r0 + 1, c0)) * fc
        v = fc === 0 && fr === 0 ? at(r0, c0) : a + (b - a) * fr
      } else {
        const row = (r: number): number => cubic(at(r, c0 - 1), at(r, c0), at(r, c0 + 1), at(r, c0 + 2), fc)
        v = cubic(row(r0 - 1), row(r0), row(r0 + 1), row(r0 + 2), fr)
        if (Number.isFinite(v)) v = Math.min(hi, Math.max(lo, v))
      }
      out[i * cols + j] = v
    }
  }
  return out
}

/** Repeat each cell `k` times along both axes (blocky cells as an image). */
export function repeatCells(g: Grid2D, k: number): Float64Array {
  const cols = g.cols * k
  const out = new Float64Array(g.rows * k * cols)
  for (let r = 0; r < g.rows * k; r++)
    for (let c = 0; c < cols; c++)
      out[r * cols + c] = g.values[Math.floor(r / k) * g.cols + Math.floor(c / k)] ?? Number.NaN
  return out
}
