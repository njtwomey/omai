// Internal: resampling of gridded values for display (corner-aligned, so grid nodes keep their values).
import type { Grid2D } from './_data.ts'

/** How a grid is resampled for display. */
export type Resampling = 'nearest' | 'bilinear' | 'bicubic'

function cubic(p0: number, p1: number, p2: number, p3: number, t: number): number {
  // Catmull-Rom: passes through p1 at t = 0 and p2 at t = 1.
  return p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)))
}

// For each output index along one axis, the source index it falls on and how far past it (corner-aligned).
function axisTable(from: number, to: number): { readonly at: Int32Array; readonly frac: Float64Array } {
  const at = new Int32Array(to)
  const frac = new Float64Array(to)
  for (let i = 0; i < to; i++) {
    const u = to === 1 ? 0 : (i * (from - 1)) / (to - 1)
    const i0 = Math.floor(u)
    at[i] = i0
    frac[i] = u - i0
  }
  return { at, frac }
}

// Catmull-Rom in two 1-D passes, columns then rows: the same sums in the same order as the 2-D kernel (a row of four
// taps at the column position, then four such rows at the row position), but each source row is interpolated once
// instead of once per output row that touches it, and the clamped index lookups are hoisted out of the inner loops.
function bicubic(g: Grid2D, rows: number, cols: number, lo: number, hi: number): Float64Array {
  const v = g.values
  const last = g.cols - 1
  const across = axisTable(g.cols, cols)
  const down = axisTable(g.rows, rows)
  const wide = new Float64Array(g.rows * cols)
  for (let r = 0; r < g.rows; r++) {
    const base = r * g.cols
    for (let j = 0; j < cols; j++) {
      const c0 = across.at[j] ?? 0
      const a = v[base + Math.min(last, Math.max(0, c0 - 1))] ?? Number.NaN
      const b = v[base + Math.min(last, c0)] ?? Number.NaN
      const c = v[base + Math.min(last, c0 + 1)] ?? Number.NaN
      const d = v[base + Math.min(last, c0 + 2)] ?? Number.NaN
      wide[r * cols + j] = cubic(a, b, c, d, across.frac[j] ?? 0)
    }
  }
  const out = new Float64Array(rows * cols)
  const bottom = g.rows - 1
  for (let i = 0; i < rows; i++) {
    const r0 = down.at[i] ?? 0
    const fr = down.frac[i] ?? 0
    const a = Math.min(bottom, Math.max(0, r0 - 1)) * cols
    const b = Math.min(bottom, r0) * cols
    const c = Math.min(bottom, r0 + 1) * cols
    const d = Math.min(bottom, r0 + 2) * cols
    for (let j = 0; j < cols; j++) {
      const x = cubic(
        wide[a + j] ?? Number.NaN,
        wide[b + j] ?? Number.NaN,
        wide[c + j] ?? Number.NaN,
        wide[d + j] ?? Number.NaN,
        fr,
      )
      out[i * cols + j] = Number.isFinite(x) ? Math.min(hi, Math.max(lo, x)) : x
    }
  }
  return out
}

/**
 * Resample a grid to `rows` by `cols` with corners aligned to the input's corner nodes. NaN propagates to the output
 * pixels it touches. Bicubic output is clamped to the input's range so ramps never overshoot.
 */
export function resample(g: Grid2D, rows: number, cols: number, method: Resampling): Float64Array {
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const v of g.values)
    if (Number.isFinite(v)) {
      lo = Math.min(lo, v)
      hi = Math.max(hi, v)
    }
  if (method === 'bicubic') return bicubic(g, rows, cols, lo, hi)
  const out = new Float64Array(rows * cols)
  const at = (r: number, c: number): number => {
    const rr = Math.min(g.rows - 1, Math.max(0, r))
    const cc = Math.min(g.cols - 1, Math.max(0, c))
    return g.values[rr * g.cols + cc] ?? Number.NaN
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
      else {
        const a = at(r0, c0) + (at(r0, c0 + 1) - at(r0, c0)) * fc
        const b = at(r0 + 1, c0) + (at(r0 + 1, c0 + 1) - at(r0 + 1, c0)) * fc
        v = fc === 0 && fr === 0 ? at(r0, c0) : a + (b - a) * fr
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
