/**
 * Example "A neural additive model, from its spline bases": the same model as the proof of concept "Generalised
 * additive model, neural" in apps/pocs. One feature is passed through a learnable B-spline, the sum of weighted basis
 * functions, and the coefficients (and here the knots) are trained by an optimiser. The smoothness penalty is zero, so
 * nothing stops the curve from following the points, and the animation `learning.gif` shows the weighted bases moving
 * and their sum coming up to the data. The dotted line is the classic GAM of the same data (REML picks its own
 * smoothing) and the dashed line the truth the data was drawn from.
 */
import { generate } from 'mdml-engine/data'
import { sineCurve } from 'mdml-engine/datasets'
import { additiveRegressor, gam, s as splineOf } from 'mdml-engine/gam'
import { key } from 'mdml-engine/random'
import { circle, line, path, rect, type Scene, scene, timeline } from 'mdml-engine/scene'
import { asTensor, fromStorage, toFlat } from 'mdml-engine/tensor'
import { Log, metricSeries } from 'mdml-engine/trace'
import { cartesian, lines, points } from 'mdml-render/charts'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Row } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'A neural additive model, from its spline bases',
  summary:
    'A learnable spline with no smoothness penalty: the weighted basis functions and their sum come up to the points as it trains (learning.gif).',
  readme: false,
  alt: 'A spline fitted with no smoothness penalty: its weighted basis functions and their sum against the data, the truth and the classic GAM, and the training loss',
}

const STEPS = 600
const KNOTS = 11
const GRID = 200

/** What the figure and the animation are drawn from. */
interface Result {
  readonly x: readonly number[]
  readonly y: readonly number[]
  readonly grid: readonly number[]
  readonly truth: readonly number[]
  readonly classic: readonly number[]
  readonly steps: readonly number[]
  /** Per shown step: the fitted curve, the weighted basis functions and the knots. */
  readonly curves: readonly (readonly number[])[]
  readonly bases: readonly (readonly (readonly number[])[])[]
  readonly knots: readonly (readonly number[])[]
  readonly loss: readonly number[]
}

export function run(): Outcome<Result> {
  const { data, truth } = generate(sineCurve, { knobs: { n: 16, noise: 0.2, design: 'sorted' }, key: key(0) })
  const x = Array.from(toFlat(asTensor(data.x)))
  const y = Array.from(toFlat(asTensor(data.y)))
  const rows = x.map((v) => [v])
  const meanY = y.reduce((a, b) => a + b, 0) / y.length
  const lo = Math.min(...x)
  const hi = Math.max(...x)
  const grid = Array.from({ length: GRID }, (_, i) => lo + ((hi - lo) * i) / (GRID - 1))

  // The model of the proof of concept: a cubic B-spline term with learnable knots, trained by an optimiser, and a
  // smoothness penalty of zero, so the only limit on the curve is the number of its bases.
  const model = additiveRegressor.with({
    lambda: 0,
    knots: { kind: 'learnable' },
    interiorKnots: KNOTS,
    degree: 3,
    gridSize: GRID,
    epochs: STEPS,
    learningRate: 0.03,
  })
  // A step budget much longer than the training keeps the session's checkpoints few, so the tape stays small.
  const training = model.session(data, { key: key(1), steps: 20000, logs: { 'train/loss': Log() } })
  training.advance()
  const steps = [0, 1, 2, 3, 5, 8, 12, 17, 24, 33, 45, 60, 80, 105, 135, 170, 215, 280, 360, 460, STEPS]
  const probed = steps.map((t) => ({
    curve: training.probe(t, 'predictor/0/term/0/curve') as { values: number[] },
    basis: training.probe(t, 'predictor/0/term/0/basis') as { functions: number[][] },
    knots: training.probe(t, 'additive/knots') as { terms?: { knots?: number[] }[] }[],
  }))

  // The truth, centred like the term (its mean over the rows is zero), and the classic fit of the same data.
  const f = (truth as { regressionFunction?: (v: never) => unknown }).regressionFunction
  const truthAt = (at: readonly number[]): number[] =>
    f === undefined
      ? at.map(() => 0)
      : Array.from(toFlat(asTensor(f(fromStorage(Float64Array.from(at), [at.length, 1], 'float64') as never) as never)))
  const truthMean = truthAt(x).reduce((a, b) => a + b, 0) / x.length
  const classicFit = gam.with({ terms: [splineOf(0, { k: 7 })] }).fit({ x: rows, y })
  const classic = Array.from(classicFit.predictTerms(grid.map((v) => [v])).data as Float64Array)

  const result: Result = {
    x,
    y: y.map((v) => v - meanY),
    grid,
    truth: truthAt(grid).map((v) => v - truthMean),
    classic,
    steps,
    curves: probed.map((p) => p.curve.values),
    bases: probed.map((p) => p.basis.functions),
    knots: probed.map((p) => p.knots[0]?.terms?.[0]?.knots ?? []),
    loss: metricSeries(training.run(), 'train/loss').values,
  }
  const gap = rmse(
    (result.curves.at(-1) ?? []).filter((_, i) => i % 1 === 0),
    result.truth,
  )
  const fitRows = rmse(
    x.map((v) => interpolate(grid, result.curves.at(-1) ?? [], v)),
    result.y,
  )
  const notes = [
    `${STEPS} steps, ${KNOTS} interior knots (learnable), smoothness penalty 0; training loss ${result.loss[0]?.toFixed(3)} at the start, ${result.loss.at(-1)?.toFixed(4)} at the end`,
    `root mean square distance of the curve from the points ${fitRows.toFixed(3)} (the noise is 0.2), from the true curve ${gap.toFixed(3)}`,
    `${steps.length} frames of the animation at steps ${steps.join(' ')}`,
  ]
  const shown = steps.map((step, i) => ({
    kind: 'scene' as const,
    step,
    scene: frame(result, i),
    caption: `step ${step}`,
  }))
  const hold = shown.slice(-1)
  const animation = timeline([...shown, ...hold, ...hold, ...hold], { fps: 6 })
  return { result, tapes: { session: training }, animation, notes }
}

const rmse = (a: readonly number[], b: readonly number[]): number =>
  Math.sqrt(a.reduce((s, v, i) => s + (v - (b[i] ?? 0)) ** 2, 0) / Math.max(1, a.length))

/** Linear interpolation of a curve sampled on `grid`. */
function interpolate(grid: readonly number[], values: readonly number[], at: number): number {
  const i = Math.max(
    1,
    grid.findIndex((g) => g >= at),
  )
  const [g0, g1] = [grid[i - 1] ?? 0, grid[i] ?? 1]
  const t = g1 === g0 ? 0 : (at - g0) / (g1 - g0)
  return (values[i - 1] ?? 0) * (1 - t) + (values[i] ?? 0) * t
}

const TRUTH = '#6b7280'
const FIT = '#d62728'
const BASIS = '#f4a582'
const CLASSIC = '#1f77b4'
const POINT = '#111827'
const EDGE = '#cbd5e1'

/**
 * One frame of the animation, in shapes only (the rasteriser draws no text): the weighted basis functions in pale
 * orange, their sum in red, the points, the truth (grey, dashed by dots) and the classic GAM in blue; the knots as ticks
 * on the bottom edge; under the plot the loss so far and a bar that fills as training goes.
 */
function frame(r: Result, i: number): Scene {
  const W = 540
  const H = 330
  const [px0, py0, pw, ph] = [10, 10, 520, 235] as const
  const lo = Math.min(...r.y, ...r.truth, ...r.classic) - 0.15
  const hi = Math.max(...r.y, ...r.truth, ...r.classic) + 0.15
  const xs = (v: number): number => px0 + ((v - (r.grid[0] ?? 0)) / ((r.grid.at(-1) ?? 1) - (r.grid[0] ?? 0))) * pw
  const ys = (v: number): number => py0 + ph - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * ph
  const curve = (values: readonly number[]): [number, number][] => values.map((v, k) => [xs(r.grid[k] ?? 0), ys(v)])
  const nodes = []
  nodes.push(rect(px0, py0, pw, ph, { fill: '#ffffff', stroke: EDGE }))
  nodes.push(line(px0, ys(0), px0 + pw, ys(0), { stroke: EDGE, dash: [3, 3] }))
  for (const basis of r.bases[i] ?? []) nodes.push(path(curve(basis), { stroke: BASIS, strokeWidth: 1.2 }))
  nodes.push(path(curve(r.truth), { stroke: TRUTH, strokeWidth: 2, dash: [2, 3] }))
  nodes.push(path(curve(r.classic), { stroke: CLASSIC, strokeWidth: 2 }))
  nodes.push(path(curve(r.curves[i] ?? []), { stroke: FIT, strokeWidth: 2.5 }))
  for (const [k, v] of r.x.entries()) nodes.push(circle(xs(v), ys(r.y[k] ?? 0), 3, { fill: POINT }))
  for (const knot of r.knots[i] ?? [])
    nodes.push(line(xs(knot), py0 + ph - 7, xs(knot), py0 + ph, { stroke: FIT, strokeWidth: 2 }))
  // The loss so far, on the axes of the whole run.
  const [lx, ly, lw, lh] = [10, 258, 520, 56] as const
  const logged = r.loss.map((v) => Math.log10(v))
  const top = Math.max(...logged)
  const bottom = Math.min(...logged)
  const upTo = Math.min(r.loss.length, (r.steps[i] ?? 0) + 1)
  const loss = logged
    .slice(0, upTo)
    .map((v, k): [number, number] => [
      lx + (k / (r.loss.length - 1)) * lw,
      ly + lh - ((v - bottom) / (top - bottom)) * lh,
    ])
  nodes.push(rect(lx, ly, lw, lh, { fill: '#ffffff', stroke: EDGE }))
  if (loss.length > 1) nodes.push(path(loss, { stroke: FIT, strokeWidth: 2 }))
  nodes.push(rect(lx, 320, lw, 4, { fill: EDGE }))
  nodes.push(rect(lx, 320, (lw * (r.steps[i] ?? 0)) / STEPS, 4, { fill: FIT }))
  return scene(W, H, nodes)
}

export function View({ result }: { result: Result }) {
  const last = result.steps.length - 1
  const fit = cartesian(
    [
      ...(result.bases[last] ?? []).map((basis, k) =>
        lines([
          { x: [...result.grid], y: [...basis], name: k === 0 ? 'weighted bases' : '', colour: BASIS, width: 1.2 },
        ]),
      ),
      lines([{ x: [...result.grid], y: [...result.truth], name: 'truth', colour: TRUTH, width: 2, dash: 'dotted' }]),
      lines([{ x: [...result.grid], y: [...result.classic], name: 'classic GAM', colour: CLASSIC, width: 2 }]),
      lines([
        {
          x: [...result.grid],
          y: [...(result.curves[last] ?? [])],
          name: 'spline, no penalty',
          colour: FIT,
          width: 2.5,
        },
      ]),
      points([{ x: [...result.x], y: [...result.y], name: 'data', colour: POINT }]),
    ],
    { x: { label: 'x' }, y: { label: 'effect' }, legend: true },
  )
  const start = cartesian(
    [
      lines([
        { x: [...result.grid], y: [...(result.curves[0] ?? [])], name: 'spline at step 0', colour: FIT, width: 2.5 },
      ]),
      points([{ x: [...result.x], y: [...result.y], name: 'data', colour: POINT }]),
    ],
    { x: { label: 'x' }, y: { label: 'effect' }, legend: false },
  )
  return (
    <Row
      alt={meta.alt}
      width={460}
      height={330}
      panels={[
        { title: 'at the start', still: start },
        { title: `after ${result.steps[last]} steps`, still: fit },
        {
          title: 'training loss',
          still: cartesian([lines([{ y: [...result.loss], colour: FIT, width: 2 }])], {
            x: { label: 'step' },
            y: { label: 'squared error', scale: 'log' },
          }),
        },
      ]}
    />
  )
}
