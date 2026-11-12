/**
 * The tensorisation experiment: the pair profile of the shared grid integrator (`message-passing/numerics/_grid.ts`, `integratePair`)
 * rewritten with tensor operations only, next to the library's scalar implementation on the same integrand, for the study in
 * `docs/probprog/ep-perf-tensor-study.md`. The tensor version follows the library's algorithm pass for pass (graded Gauss-Legendre
 * panels split at the kinks, log weights, the shift by the maximum, normalisation, marginals, the zoom of each window onto the mass,
 * the moments) with the same arithmetic in the same order, so the two can be compared number for number:
 *
 * - the nodes are made with `arange`, `pow`, `concat`, `flip` and broadcasting (an outer sum of panel start and panel offset);
 * - the log density of the cavities is a tensor expression of the nodes;
 * - the integrand is one broadcast expression over the whole `n0 x n1` node grid (the column form's "once per node of the second axis"
 *   is a `[1, n1]` factor);
 * - the reduction is `max`, `exp`, `sum` over the grid and over each axis.
 *
 * Reported: the largest difference in the log mass, the means and the variances, whether the numbers are bit-identical, the time per
 * call at the current node counts and at two and four times the panels, and the memory allocated per call (measured in a child
 * process with a semispace large enough that no collection runs). Not part of the library.
 *
 *   nice -n 15 node --conditions=source --experimental-transform-types --no-warnings scripts/perf/tensor-grid-study.ts [--json <file>]
 */
import { execFileSync } from 'node:child_process'
import { writeFileSync } from 'node:fs'
import {
  type GridAxis,
  type GridProfile,
  gammaAxis,
  gaussianAxis,
  gridProfiles,
  integratePair,
} from '../../packages/engine/src/lib/message-passing/numerics/_grid.ts'
import { legendreRule } from '../../packages/engine/src/lib/message-passing/numerics/_quadrature.ts'
import { lgammaOf } from '../../packages/engine/src/lib/message-passing/numerics/_scalar.ts'
import { gammaLogWindow, windowProfiles } from '../../packages/engine/src/lib/message-passing/numerics/_windows.ts'
import {
  abs,
  add,
  arange,
  concat,
  div,
  exp,
  flip,
  log,
  max,
  mul,
  neg,
  pow,
  reshape,
  sliceAxis,
  square,
  sub,
  sum,
  tensor,
  toFlat,
  type Value,
  zeros,
} from '../../packages/engine/src/lib/tensor/index.ts'
import { summarise } from '../ep-perf-stats.ts'
import { Y } from './grid-cases.ts'
import { bench, calibrate, keep, now } from './timing.ts'

// ----- the tensor version of the pair integrator -----

/** A cavity axis for the tensor integrator: the window and kinks as the library's, the log density as a tensor expression of the nodes. */
interface AxisT {
  readonly lo: number
  readonly hi: number
  readonly kinks: readonly number[]
  readonly logWeight: (x: Value) => Value
}

/** The Gaussian cavity's axis (the library's `gaussianAxis`), the density in tensor operations. */
function gaussianAxisT(mean: number, variance: number, kinks: readonly number[]): AxisT {
  const a = gaussianAxis(mean, variance, kinks)
  const c = Math.log(2 * Math.PI * variance)
  return { lo: a.lo, hi: a.hi, kinks, logWeight: (x) => mul(-0.5, add(c, div(square(sub(x, mean)), variance))) }
}

/** The gamma cavity's axis in the log coordinate (the library's `gammaAxis`). */
function gammaAxisT(shape: number, rate: number, kinks: readonly number[]): AxisT {
  const { lo, hi } = gammaLogWindow(shape, rate, windowProfiles.gridGamma)
  const logNorm = lgammaOf(shape) - shape * Math.log(rate)
  return { lo, hi, kinks, logWeight: (t) => sub(sub(mul(shape, t), mul(rate, exp(t))), logNorm) }
}

/** The offsets 0, 2^-(n-1), ..., 1/4, 1/2 of a stretch of n panels. */
const gradedT = (n: number): Value => concat([zeros([1]), pow(2, neg(sub(n - 1, arange(n - 1))))])

/** The panel boundaries of [a, b], `count` panels: uniform, graded geometrically towards an end that is a kink. */
function boundariesT(a: number, b: number, count: number, kinkLeft: boolean, kinkRight: boolean): Value {
  const len = b - a
  if (kinkLeft && kinkRight) {
    const n = Math.max(2, Math.ceil(count / 2))
    const left = add(a, mul(len / 2, gradedT(n)))
    const right = flip(sub(b, mul(len / 2, gradedT(n))))
    return concat([left, right])
  }
  if (kinkLeft) return concat([add(a, mul(len, gradedT(count))), tensor([b])])
  if (kinkRight) return concat([tensor([a]), flip(sub(b, mul(len, gradedT(count))))])
  return add(a, div(mul(len, arange(count + 1)), count))
}

/** The nodes and weights of a panelled rule on [lo, hi], split at the kinks inside it: a broadcast of panel starts against rule offsets. */
function gradedPanelsT(
  lo: number,
  hi: number,
  kinks: readonly number[],
  count: number,
  order: number,
): { readonly x: Value; readonly w: Value } {
  const cuts = [lo, ...kinks.filter((k) => k > lo && k < hi).sort((p, q) => p - q), hi]
  const rule = legendreRule(order)
  const rx = tensor(Array.from(rule.x))
  const rw = tensor(Array.from(rule.w))
  const unit = mul(add(rx, 1), 0.5)
  const half = mul(rw, 0.5)
  const xs: Value[] = []
  const ws: Value[] = []
  for (let s = 0; s + 1 < cuts.length; s++) {
    const a = cuts[s] as number
    const b = cuts[s + 1] as number
    const n = Math.max(6, Math.round((count * (b - a)) / (hi - lo)))
    const edges = boundariesT(a, b, n, s > 0, s + 2 < cuts.length)
    const left = reshape(sliceAxis(edges, 0, 0, n), [n, 1])
    const h = reshape(sub(sliceAxis(edges, 0, 1, n + 1), sliceAxis(edges, 0, 0, n)), [n, 1])
    xs.push(reshape(add(left, mul(reshape(unit, [1, order]), h)), [-1]))
    ws.push(reshape(mul(reshape(half, [1, order]), h), [-1]))
  }
  return { x: concat(xs), w: concat(ws) }
}

/** The result of the tensor integral: the log mass and the mean and variance of each coordinate. */
interface PairT {
  readonly logMass: number
  readonly mean: readonly [number, number]
  readonly variance: readonly [number, number]
}

/** Seconds-free counters of the last call: how many passes it made and the milliseconds spent building nodes and evaluating the grid. */
export const phases = { passes: 0, buildMs: 0, evalMs: 0, zoomMs: 0 }

const ZOOM_FLOOR = 1e-9
const ZOOM_SHRINK = 0.7

/**
 * `integratePair` with tensor operations: the same passes over the same windows with the same arithmetic.
 *
 * @param axes - The two cavities.
 * @param column - The log of the integrand of the nodes `x0` (a `[n0, 1]` tensor) and `x1` (a `[1, n1]` tensor), broadcast to `[n0, n1]`.
 * @param profile - The grid profile (the library's `pair`, or a scaled copy).
 * @returns The log mass and the moments.
 */
function integratePairT(
  axes: readonly [AxisT, AxisT],
  column: (x0: Value, x1: Value) => Value,
  profile: GridProfile,
): PairT {
  let window: [number, number][] = axes.map((a): [number, number] => [a.lo, a.hi])
  let fine = false
  phases.passes = phases.buildMs = phases.evalMs = phases.zoomMs = 0
  let last: { logMass: number; x: [Value, Value]; m: [Value, Value] } | undefined
  for (let pass = 0; pass < profile.passes; pass++) {
    const shape = fine || pass === profile.passes - 1 ? profile.fine : profile.coarse
    phases.passes += 1
    const t0 = now()
    const grids = axes.map((a, i) => {
      const [lo, hi] = window[i] as [number, number]
      const g = gradedPanelsT(lo, hi, a.kinks, shape.panels, shape.order)
      return { x: g.x, logw: add(log(g.w), a.logWeight(g.x)) }
    })
    const g0 = grids[0] as { x: Value; logw: Value }
    const g1 = grids[1] as { x: Value; logw: Value }
    const n0 = (g0.x as { shape: readonly number[] }).shape[0] as number
    const n1 = (g1.x as { shape: readonly number[] }).shape[0] as number
    const t1 = now()
    const x0 = reshape(g0.x, [n0, 1])
    const x1 = reshape(g1.x, [1, n1])
    // The log weight of every node: the two cavities' log weights and the integrand, one broadcast expression.
    const logw = add(add(reshape(g0.logw, [n0, 1]), reshape(g1.logw, [1, n1])), column(x0, x1))
    // Shift by the largest, exponentiate, normalise by the total.
    const top = max(logw)
    const weights = exp(sub(logw, top))
    const total = sum(weights)
    const w = div(weights, total)
    const m0 = sum(w, 1)
    const m1 = sum(w, 0)
    const logMass = (toFlat(top)[0] as number) + Math.log(toFlat(total)[0] as number)
    if (!Number.isFinite(toFlat(top)[0] as number)) throw new Error('no mass')
    last = { logMass, x: [g0.x, g1.x], m: [m0, m1] }
    const t2 = now()
    phases.buildMs += t1 - t0
    phases.evalMs += t2 - t1
    // Zoom each window onto the nodes that carry weight (control flow on the marginals, small vectors).
    let shrunk = false
    const next = axes.map((a, i): [number, number] => {
      const [lo, hi] = window[i] as [number, number]
      const x = toFlat((grids[i] as { x: Value }).x)
      const m = toFlat(i === 0 ? m0 : m1)
      const peak = m.reduce((p, q) => Math.max(p, q), 0)
      let from = 0
      while (from < x.length && (m[from] as number) < ZOOM_FLOOR * peak) from++
      let to = x.length - 1
      while (to > from && (m[to] as number) < ZOOM_FLOOR * peak) to--
      const spacing = (hi - lo) / x.length
      const nlo = Math.max(a.lo, (x[Math.max(from - 1, 0)] as number) - spacing)
      const nhi = Math.min(a.hi, (x[Math.min(to + 1, x.length - 1)] as number) + spacing)
      if (!(nhi > nlo)) return [lo, hi]
      if (nhi - nlo < ZOOM_SHRINK * (hi - lo)) shrunk = true
      return [nlo, nhi]
    })
    phases.zoomMs += now() - t2
    if (!shrunk) {
      if (fine || shape === profile.fine) break
      fine = true
      continue
    }
    window = next
  }
  const r = last as NonNullable<typeof last>
  const moments = (x: Value, m: Value): { mean: number; variance: number } => {
    const mean = toFlat(sum(mul(m, x)))[0] as number
    const variance = toFlat(sum(mul(m, square(sub(x, mean)))))[0] as number
    return { mean, variance }
  }
  const s0 = moments(r.x[0], r.m[0])
  const s1 = moments(r.x[1], r.m[1])
  return { logMass: r.logMass, mean: [s0.mean, s1.mean], variance: [s0.variance, s1.variance] }
}

// ----- the two implementations on the same integrand -----

/** The Laplace likelihood `-t - exp(-t) |y - x0|` of `grid-cases.ts`, as a broadcast tensor expression. */
const laplaceT = (x0: Value, x1: Value): Value => sub(neg(x1), mul(exp(neg(x1)), abs(sub(Y, x0))))

/** The profile `pair` with its panels multiplied (the library's own object is scaled for the call and restored). */
function scaled(profile: GridProfile, k: number): GridProfile {
  return {
    ...profile,
    coarse: { ...profile.coarse, panels: profile.coarse.panels * k },
    fine: { ...profile.fine, panels: profile.fine.panels * k },
  }
}

/** Runs the library's `integratePair` with the pair profile scaled by `k` (the profile table is plain data; restored afterwards). */
function scalarCall(k: number): {
  logMass: number
  mean: readonly [number, number]
  variance: readonly [number, number]
  weights: Float64Array
} {
  const table = gridProfiles as unknown as { pair: GridProfile }
  const original = table.pair
  table.pair = scaled(original, k)
  try {
    const axes: [GridAxis, GridAxis] = [gaussianAxis(0.4, 1.0, [Y]), gammaAxis(4, 3)]
    return integratePair(axes, (t) => {
      const s = Math.exp(-t)
      return (x0) => -t - s * Math.abs(Y - x0)
    })
  } finally {
    table.pair = original
  }
}

function tensorCall(k: number): PairT {
  const axes: [AxisT, AxisT] = [gaussianAxisT(0.4, 1.0, [Y]), gammaAxisT(4, 3, [])]
  return integratePairT(axes, laplaceT, scaled(gridProfiles.pair, k))
}

interface Result {
  readonly scale: number
  readonly nodes: number
  readonly maxAbsLogMass: number
  readonly maxAbsMean: number
  readonly maxAbsVariance: number
  readonly bitIdentical: boolean
  readonly scalarUs: number
  readonly scalarSpread: number
  readonly tensorUs: number
  readonly tensorSpread: number
  readonly ratio: number
}

function study(): Result[] {
  const results: Result[] = []
  for (const k of [1, 2, 4]) {
    const a = scalarCall(k)
    const b = tensorCall(k)
    const diff = (x: number, y: number): number => Math.abs(x - y)
    const maxAbsLogMass = diff(a.logMass, b.logMass)
    const maxAbsMean = Math.max(diff(a.mean[0], b.mean[0]), diff(a.mean[1], b.mean[1]))
    const maxAbsVariance = Math.max(diff(a.variance[0], b.variance[0]), diff(a.variance[1], b.variance[1]))
    const bitIdentical =
      Object.is(a.logMass, b.logMass) &&
      a.mean.every((v, i) => Object.is(v, b.mean[i])) &&
      a.variance.every((v, i) => Object.is(v, b.variance[i]))
    const nodes = a.weights.length
    const s = summarise(bench(() => scalarCall(k), 9, 30))
    const t = summarise(bench(() => tensorCall(k), 9, 30))
    results.push({
      scale: k,
      nodes,
      maxAbsLogMass,
      maxAbsMean,
      maxAbsVariance,
      bitIdentical,
      scalarUs: s.median,
      scalarSpread: s.spread,
      tensorUs: t.median,
      tensorSpread: t.spread,
      ratio: t.median / s.median,
    })
  }
  return results
}

// ----- allocation, in a child process with no collection during the measurement -----

interface Alloc {
  readonly heapKbPerCall: number
  readonly bufferKbPerCall: number
}

/** The memory one call allocates: heap and typed-array storage grown over `calls` calls with the collector held off (called in the child). */
function allocation(which: 'scalar' | 'tensor', scale: number, calls: number): Alloc {
  const run = which === 'scalar' ? () => scalarCall(scale) : () => tensorCall(scale)
  for (let i = 0; i < 3; i++) keep(run().logMass)
  const gc = (globalThis as unknown as { gc?: () => void }).gc
  gc?.()
  const before = process.memoryUsage()
  for (let i = 0; i < calls; i++) keep(run().logMass)
  const after = process.memoryUsage()
  return {
    heapKbPerCall: (after.heapUsed - before.heapUsed) / 1024 / calls,
    bufferKbPerCall: (after.arrayBuffers - before.arrayBuffers) / 1024 / calls,
  }
}

const argv = process.argv.slice(2)
const childIndex = argv.indexOf('--alloc')
if (childIndex >= 0) {
  const [which, scale] = [argv[childIndex + 1] as 'scalar' | 'tensor', Number(argv[childIndex + 2])]
  process.stdout.write(`${JSON.stringify(allocation(which, scale, 20))}\n`)
} else {
  const calStart = calibrate(3)
  const results = study()
  const calEnd = calibrate(3)
  const breakdown = [1, 2, 4].map((k) => {
    tensorCall(k)
    const ms: number[][] = [[], [], []]
    let passes = 0
    for (let i = 0; i < 15; i++) {
      tensorCall(k)
      ms[0]?.push(phases.buildMs)
      ms[1]?.push(phases.evalMs)
      ms[2]?.push(phases.zoomMs)
      passes = phases.passes
    }
    return {
      scale: k,
      passes,
      buildUs: summarise(ms[0] as number[]).median * 1000,
      evalUs: summarise(ms[1] as number[]).median * 1000,
      zoomUs: summarise(ms[2] as number[]).median * 1000,
    }
  })
  for (const b of breakdown)
    process.stdout.write(
      `tensor ${b.scale}x: ${b.passes} passes, nodes and cavity densities ${b.buildUs.toFixed(0)} us, grid expression and reductions ${b.evalUs.toFixed(0)} us, zoom ${b.zoomUs.toFixed(0)} us\n`,
    )
  for (const r of results)
    process.stdout.write(
      `scale ${r.scale}x (${r.nodes} final nodes): |dlogMass| ${r.maxAbsLogMass.toExponential(2)}, |dmean| ${r.maxAbsMean.toExponential(2)}, |dvar| ${r.maxAbsVariance.toExponential(2)}, bit-identical ${r.bitIdentical}; scalar ${r.scalarUs.toFixed(0)} us (spread ${(100 * r.scalarSpread).toFixed(0)}%), tensor ${r.tensorUs.toFixed(0)} us (spread ${(100 * r.tensorSpread).toFixed(0)}%), ratio ${r.ratio.toFixed(2)}\n`,
    )
  const allocs: Record<string, Alloc> = {}
  for (const k of [1, 2, 4])
    for (const which of ['scalar', 'tensor'] as const) {
      const text = execFileSync(
        process.execPath,
        [
          '--expose-gc',
          '--min-semi-space-size=256',
          '--max-semi-space-size=256',
          '--conditions=source',
          '--experimental-transform-types',
          '--no-warnings',
          import.meta.filename,
          '--alloc',
          which,
          String(k),
        ],
        { encoding: 'utf8' },
      )
      allocs[`${which}/${k}`] = JSON.parse(text) as Alloc
      process.stdout.write(`allocation ${which} ${k}x: ${text}`)
    }
  const json = argv.indexOf('--json')
  if (json >= 0)
    writeFileSync(
      argv[json + 1] as string,
      JSON.stringify({ results, breakdown, allocs, calibration: { start: calStart, end: calEnd } }, null, 1),
    )
}
