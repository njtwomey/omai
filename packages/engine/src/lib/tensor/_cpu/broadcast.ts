import type { CompareKernel } from '../backend.ts'
import type { DType, Storage } from '../dtype.ts'
import type { Shape } from '../shape.ts'

// Broadcast fast paths of the elementwise CPU kernels: a plan that describes how the operands of a broadcast map onto the
// output without an index map (the general path, `broadcastIndex`, writes two output-sized `Int32Array`s and calls the
// operation as a closure per element), and the row loops that run a plan. The common layouts are all covered by one plan:
// an outer product `[n,1] + [1,m]`, a row or a column against a matrix, a trailing-axis operand `[..., k]` against `[k]`, a
// leading-axis operand `[a, 1, 1]` against the full shape, and the rank-2 and rank-3 grids. A plan exists when the
// output, with its axes of extent 1 dropped and adjacent axes that every operand treats alike merged, has at most four axes;
// otherwise the caller keeps the general path. The loops evaluate the same IEEE operation per element in the same order, so
// results are bit-identical to the general path (the property tests compare them with `Object.is`).

/** How the operands of a broadcast walk the output: four extents (outermost first) and, per operand, the stride on each. */
export interface BroadcastPlan {
  readonly extents: readonly [number, number, number, number]
  /** Per operand and axis: 0 where the operand repeats along the axis, otherwise the (contiguous) step in its data. */
  readonly strides: readonly (readonly [number, number, number, number])[]
}

type Axes4 = [number, number, number, number]

/**
 * Plan a broadcast of `operands` to `shape` (the broadcast shape of them), or `null` when it needs more than four axes.
 * Axes of extent 1 are dropped; adjacent axes that every operand either reads in full or repeats are merged. The
 * innermost axis of a plan is read with stride 1 (in full) or 0 (repeated) by every operand, and at least one operand reads it in full.
 */
export function planBroadcast(shape: Shape, operands: readonly Shape[]): BroadcastPlan | null {
  const rank = shape.length
  const count = operands.length
  // The retained axes with a bit mask of the operands that read each in full (bit k: operand k); equal masks next to each other merge.
  const extents: number[] = []
  const masks: number[] = []
  for (let d = 0; d < rank; d++) {
    const n = shape[d] as number
    if (n === 1) continue
    let mask = 0
    for (let k = 0; k < count; k++) {
      const s = operands[k] as Shape
      if ((s[d - (rank - s.length)] ?? 1) === n) mask |= 1 << k
    }
    const last = extents.length - 1
    if (last >= 0 && masks[last] === mask) extents[last] = (extents[last] as number) * n
    else {
      extents.push(n)
      masks.push(mask)
    }
  }
  if (extents.length === 0) {
    extents.push(1)
    masks.push((1 << count) - 1)
  }
  if (extents.length > 4) return null
  const pad = 4 - extents.length
  const ext: Axes4 = [1, 1, 1, 1]
  for (let i = 0; i < extents.length; i++) ext[pad + i] = extents[i] as number
  const strides: Axes4[] = []
  for (let k = 0; k < count; k++) {
    const stride: Axes4 = [0, 0, 0, 0]
    let step = 1
    for (let d = 3; d >= pad; d--) {
      if (((masks[d - pad] as number) >> k) & 1) {
        stride[d] = step
        step *= ext[d] as number
      }
    }
    strides.push(stride)
  }
  return { extents: ext, strides }
}

type Data = Storage<DType>

/** A loop over one output row of `n` elements starting at `o`; `sa` and `sb` (0 or 1) say whether each operand advances. */
export type Row2 = (
  out: Data,
  o: number,
  ad: Data,
  ia: number,
  sa: number,
  bd: Data,
  ib: number,
  sb: number,
  n: number,
) => void

/** The row loop of a ternary operation (`where`): each operand has its own stride (0 or 1). */
export type Row3 = (
  out: Data,
  o: number,
  cd: Data,
  ic: number,
  sc: number,
  ad: Data,
  ia: number,
  sa: number,
  bd: Data,
  ib: number,
  sb: number,
  n: number,
) => void

/** Run a row loop over the plan of two operands (output rows are contiguous). */
export function runRows2(row: Row2, out: Data, a: Data, b: Data, plan: BroadcastPlan): void {
  const [e0, e1, e2, n] = plan.extents
  const [a0, a1, a2, a3] = plan.strides[0] as Axes4
  const [b0, b1, b2, b3] = plan.strides[1] as Axes4
  let o = 0
  for (let i0 = 0; i0 < e0; i0++)
    for (let i1 = 0; i1 < e1; i1++)
      for (let i2 = 0; i2 < e2; i2++) {
        row(out, o, a, i0 * a0 + i1 * a1 + i2 * a2, a3, b, i0 * b0 + i1 * b1 + i2 * b2, b3, n)
        o += n
      }
}

/** Run a row loop over the plan of three operands. */
export function runRows3(row: Row3, out: Data, c: Data, a: Data, b: Data, plan: BroadcastPlan): void {
  const [e0, e1, e2, n] = plan.extents
  const [c0, c1, c2, c3] = plan.strides[0] as Axes4
  const [a0, a1, a2, a3] = plan.strides[1] as Axes4
  const [b0, b1, b2, b3] = plan.strides[2] as Axes4
  let o = 0
  for (let i0 = 0; i0 < e0; i0++)
    for (let i1 = 0; i1 < e1; i1++)
      for (let i2 = 0; i2 < e2; i2++) {
        row(
          out,
          o,
          c,
          i0 * c0 + i1 * c1 + i2 * c2,
          c3,
          a,
          i0 * a0 + i1 * a1 + i2 * a2,
          a3,
          b,
          i0 * b0 + i1 * b1 + i2 * b2,
          b3,
          n,
        )
        o += n
      }
}

/** Copy one operand to the output through its plan: rows that repeat an element are filled, the others copied. */
export function runCopy(out: Data, x: Data, plan: BroadcastPlan): void {
  const [e0, e1, e2, n] = plan.extents
  const [s0, s1, s2, sn] = plan.strides[0] as Axes4
  let o = 0
  for (let i0 = 0; i0 < e0; i0++)
    for (let i1 = 0; i1 < e1; i1++)
      for (let i2 = 0; i2 < e2; i2++) {
        const from = i0 * s0 + i1 * s1 + i2 * s2
        if (sn === 0) out.fill(x[from] as number, o, o + n)
        else for (let j = 0; j < n; j++) out[o + j] = x[from + j] as number
        o += n
      }
}

/** Run an arbitrary function over the plan of two operands (no index maps, but one call per element). */
export function runFn2(fn: (x: number, y: number) => number, out: Data, a: Data, b: Data, plan: BroadcastPlan): void {
  const [e0, e1, e2, n] = plan.extents
  const [a0, a1, a2, sa] = plan.strides[0] as Axes4
  const [b0, b1, b2, sb] = plan.strides[1] as Axes4
  let o = 0
  for (let i0 = 0; i0 < e0; i0++)
    for (let i1 = 0; i1 < e1; i1++)
      for (let i2 = 0; i2 < e2; i2++) {
        const ia = i0 * a0 + i1 * a1 + i2 * a2
        const ib = i0 * b0 + i1 * b1 + i2 * b2
        for (let j = 0; j < n; j++) out[o + j] = fn(a[ia + j * sa] as number, b[ib + j * sb] as number)
        o += n
      }
}

/** C99 and numpy: `1 ** y` and `x ** 0` are 1 even for NaN and Infinity (JavaScript's `**` gives NaN there). */
export function powKernel(a: number, b: number): number {
  return a === 1 || b === 0 || (a === -1 && Math.abs(b) === Infinity) ? 1 : a ** b
}

// One loop per operation, written out (a closure per element would be megamorphic and keep the loops from being compiled
// to straight-line code). Each takes the three layouts of an inner row: both operands advance, only the second, only the
// first. A row where neither advances never occurs (the output extent would be 1 and is dropped by the plan).

export const arithmeticRows: Record<'add' | 'sub' | 'mul' | 'div' | 'max' | 'min' | 'pow', Row2> = {
  add: (out, o, ad, ia, sa, bd, ib, sb, n) => {
    if (sa === 1 && sb === 1) for (let j = 0; j < n; j++) out[o + j] = (ad[ia + j] as number) + (bd[ib + j] as number)
    else if (sa === 0) for (let j = 0, x = ad[ia] as number; j < n; j++) out[o + j] = x + (bd[ib + j] as number)
    else for (let j = 0, y = bd[ib] as number; j < n; j++) out[o + j] = (ad[ia + j] as number) + y
  },
  sub: (out, o, ad, ia, sa, bd, ib, sb, n) => {
    if (sa === 1 && sb === 1) for (let j = 0; j < n; j++) out[o + j] = (ad[ia + j] as number) - (bd[ib + j] as number)
    else if (sa === 0) for (let j = 0, x = ad[ia] as number; j < n; j++) out[o + j] = x - (bd[ib + j] as number)
    else for (let j = 0, y = bd[ib] as number; j < n; j++) out[o + j] = (ad[ia + j] as number) - y
  },
  mul: (out, o, ad, ia, sa, bd, ib, sb, n) => {
    if (sa === 1 && sb === 1) for (let j = 0; j < n; j++) out[o + j] = (ad[ia + j] as number) * (bd[ib + j] as number)
    else if (sa === 0) for (let j = 0, x = ad[ia] as number; j < n; j++) out[o + j] = x * (bd[ib + j] as number)
    else for (let j = 0, y = bd[ib] as number; j < n; j++) out[o + j] = (ad[ia + j] as number) * y
  },
  div: (out, o, ad, ia, sa, bd, ib, sb, n) => {
    if (sa === 1 && sb === 1) for (let j = 0; j < n; j++) out[o + j] = (ad[ia + j] as number) / (bd[ib + j] as number)
    else if (sa === 0) for (let j = 0, x = ad[ia] as number; j < n; j++) out[o + j] = x / (bd[ib + j] as number)
    else for (let j = 0, y = bd[ib] as number; j < n; j++) out[o + j] = (ad[ia + j] as number) / y
  },
  // Math.max and Math.min propagate NaN and order the zeros, which is what the general path's guarded call computes.
  max: (out, o, ad, ia, sa, bd, ib, sb, n) => {
    if (sa === 1 && sb === 1)
      for (let j = 0; j < n; j++) out[o + j] = Math.max(ad[ia + j] as number, bd[ib + j] as number)
    else if (sa === 0) for (let j = 0, x = ad[ia] as number; j < n; j++) out[o + j] = Math.max(x, bd[ib + j] as number)
    else for (let j = 0, y = bd[ib] as number; j < n; j++) out[o + j] = Math.max(ad[ia + j] as number, y)
  },
  min: (out, o, ad, ia, sa, bd, ib, sb, n) => {
    if (sa === 1 && sb === 1)
      for (let j = 0; j < n; j++) out[o + j] = Math.min(ad[ia + j] as number, bd[ib + j] as number)
    else if (sa === 0) for (let j = 0, x = ad[ia] as number; j < n; j++) out[o + j] = Math.min(x, bd[ib + j] as number)
    else for (let j = 0, y = bd[ib] as number; j < n; j++) out[o + j] = Math.min(ad[ia + j] as number, y)
  },
  pow: (out, o, ad, ia, sa, bd, ib, sb, n) => {
    if (sa === 1 && sb === 1)
      for (let j = 0; j < n; j++) out[o + j] = powKernel(ad[ia + j] as number, bd[ib + j] as number)
    else if (sa === 0) for (let j = 0, x = ad[ia] as number; j < n; j++) out[o + j] = powKernel(x, bd[ib + j] as number)
    else for (let j = 0, y = bd[ib] as number; j < n; j++) out[o + j] = powKernel(ad[ia + j] as number, y)
  },
}

export const compareRows: Record<CompareKernel, Row2> = {
  eq: (out, o, ad, ia, sa, bd, ib, sb, n) => {
    if (sa === 1 && sb === 1) for (let j = 0; j < n; j++) out[o + j] = ad[ia + j] === bd[ib + j] ? 1 : 0
    else if (sa === 0) for (let j = 0, x = ad[ia]; j < n; j++) out[o + j] = x === bd[ib + j] ? 1 : 0
    else for (let j = 0, y = bd[ib]; j < n; j++) out[o + j] = ad[ia + j] === y ? 1 : 0
  },
  ne: (out, o, ad, ia, sa, bd, ib, sb, n) => {
    if (sa === 1 && sb === 1) for (let j = 0; j < n; j++) out[o + j] = ad[ia + j] !== bd[ib + j] ? 1 : 0
    else if (sa === 0) for (let j = 0, x = ad[ia]; j < n; j++) out[o + j] = x !== bd[ib + j] ? 1 : 0
    else for (let j = 0, y = bd[ib]; j < n; j++) out[o + j] = ad[ia + j] !== y ? 1 : 0
  },
  lt: (out, o, ad, ia, sa, bd, ib, sb, n) => {
    if (sa === 1 && sb === 1)
      for (let j = 0; j < n; j++) out[o + j] = (ad[ia + j] as number) < (bd[ib + j] as number) ? 1 : 0
    else if (sa === 0) for (let j = 0, x = ad[ia] as number; j < n; j++) out[o + j] = x < (bd[ib + j] as number) ? 1 : 0
    else for (let j = 0, y = bd[ib] as number; j < n; j++) out[o + j] = (ad[ia + j] as number) < y ? 1 : 0
  },
  le: (out, o, ad, ia, sa, bd, ib, sb, n) => {
    if (sa === 1 && sb === 1)
      for (let j = 0; j < n; j++) out[o + j] = (ad[ia + j] as number) <= (bd[ib + j] as number) ? 1 : 0
    else if (sa === 0)
      for (let j = 0, x = ad[ia] as number; j < n; j++) out[o + j] = x <= (bd[ib + j] as number) ? 1 : 0
    else for (let j = 0, y = bd[ib] as number; j < n; j++) out[o + j] = (ad[ia + j] as number) <= y ? 1 : 0
  },
  gt: (out, o, ad, ia, sa, bd, ib, sb, n) => {
    if (sa === 1 && sb === 1)
      for (let j = 0; j < n; j++) out[o + j] = (ad[ia + j] as number) > (bd[ib + j] as number) ? 1 : 0
    else if (sa === 0) for (let j = 0, x = ad[ia] as number; j < n; j++) out[o + j] = x > (bd[ib + j] as number) ? 1 : 0
    else for (let j = 0, y = bd[ib] as number; j < n; j++) out[o + j] = (ad[ia + j] as number) > y ? 1 : 0
  },
  ge: (out, o, ad, ia, sa, bd, ib, sb, n) => {
    if (sa === 1 && sb === 1)
      for (let j = 0; j < n; j++) out[o + j] = (ad[ia + j] as number) >= (bd[ib + j] as number) ? 1 : 0
    else if (sa === 0)
      for (let j = 0, x = ad[ia] as number; j < n; j++) out[o + j] = x >= (bd[ib + j] as number) ? 1 : 0
    else for (let j = 0, y = bd[ib] as number; j < n; j++) out[o + j] = (ad[ia + j] as number) >= y ? 1 : 0
  },
}

/**
 * `where`: the layouts of a condition in full (the usual case) get a loop each, with a repeated operand read once; the rest
 * (a repeated condition) take the strided loop.
 */
export const whereRow: Row3 = (out, o, cd, ic, sc, ad, ia, sa, bd, ib, sb, n) => {
  if (sc === 1) {
    if (sa === 1 && sb === 1)
      for (let j = 0; j < n; j++) out[o + j] = cd[ic + j] ? (ad[ia + j] as number) : (bd[ib + j] as number)
    else if (sa === 0 && sb === 1)
      for (let j = 0, x = ad[ia] as number; j < n; j++) out[o + j] = cd[ic + j] ? x : (bd[ib + j] as number)
    else if (sa === 1)
      for (let j = 0, y = bd[ib] as number; j < n; j++) out[o + j] = cd[ic + j] ? (ad[ia + j] as number) : y
    else for (let j = 0, x = ad[ia] as number, y = bd[ib] as number; j < n; j++) out[o + j] = cd[ic + j] ? x : y
  } else for (let j = 0; j < n; j++) out[o + j] = cd[ic] ? (ad[ia + j * sa] as number) : (bd[ib + j * sb] as number)
}
