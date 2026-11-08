/**
 * Example "A neural additive model, from its spline bases": the model of the proof of concept "Generalised additive
 * model, neural" in apps/pocs, with its data (a noisy sine curve) and its picture. One feature goes through a cubic
 * B-spline, a sum of weighted basis functions, and the coefficients are trained by an optimiser with a smoothness
 * penalty of zero. The animation `learning.gif` shows the weighted bases (dotted) and their sum (thick) coming up to the
 * points. The dashed line is the truth the data was drawn from, the thin dark line the classic GAM (REML chooses its
 * smoothing), and the dots on the bottom edge the knots.
 */
import { generate } from 'mdml-engine/data'
import { sineCurve } from 'mdml-engine/datasets'
import { additiveRegressor, gam, s as splineOf } from 'mdml-engine/gam'
import { key } from 'mdml-engine/random'
import { circle, line, path, rect, type Scene, type SceneNode, scene, timeline } from 'mdml-engine/scene'
import { asTensor, fromStorage, toFlat } from 'mdml-engine/tensor'
import { Log, metricSeries } from 'mdml-engine/trace'
import { cartesian, lines, points } from 'mdml-render/charts'
import { MUTED, mix, PAPER, tab } from '../_shared/colours.ts'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'
import { pixelText, pixelTextWidth } from '../_shared/pixel-text.ts'

export const meta: ExampleModule['meta'] = {
  title: 'A neural additive model, from its spline bases',
  summary:
    'A learnable spline with no smoothness penalty: the weighted basis functions and their sum come up to the points as it trains (learning.gif).',
  readme: false,
  alt: 'A spline fitted with no smoothness penalty: its weighted basis functions and their sum against the data, the truth and the classic GAM, and the training loss',
}

const STEPS = 200
const KNOTS = 8
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
  const { data, truth } = generate(sineCurve, { knobs: { n: 100, noise: 0.15, design: 'grid' }, key: key(0) })
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
    knots: { kind: 'uniform' },
    interiorKnots: KNOTS,
    degree: 3,
    gridSize: GRID,
    epochs: STEPS,
    learningRate: 0.05,
  })
  // A step budget much longer than the training keeps the session's checkpoints few, so the tape stays small.
  const training = model.session(data, { key: key(1), steps: 20000, logs: { 'train/loss': Log() } })
  training.advance()
  const steps = [0, 1, 2, 3, 5, 8, 12, 17, 24, 33, 45, 60, 80, 105, 135, 170, STEPS]
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
  const classicFit = gam.with({ terms: [splineOf(0, { k: KNOTS + 4 })] }).fit({ x: rows, y })
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
    `${STEPS} steps, ${KNOTS} interior knots, smoothness penalty 0; training loss ${result.loss[0]?.toFixed(3)} at the start, ${result.loss.at(-1)?.toFixed(4)} at the end`,
    `root mean square distance of the curve from the points ${fitRows.toFixed(3)} (the noise is 0.15), from the true curve ${gap.toFixed(3)}`,
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
  const found = grid.findIndex((g) => g >= at)
  const i = found < 0 ? grid.length - 1 : Math.max(1, found)
  const [g0, g1] = [grid[i - 1] ?? 0, grid[i] ?? 1]
  const t = g1 === g0 ? 0 : (at - g0) / (g1 - g0)
  return (values[i - 1] ?? 0) * (1 - t) + (values[i] ?? 0) * t
}

const TRUTH = 'muted'
const FIT = 'tab1'
const BASIS = mix(tab(1), PAPER, 0.55)
const CLASSIC = 'fg'
const POINT = 'muted'

/** The ticks of an axis from `lo` to `hi`, about `count` of them at round numbers. */
function ticks(lo: number, hi: number, count: number): number[] {
  const raw = (hi - lo) / count
  const magnitude = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 5, 10].map((m) => m * magnitude).find((v) => v >= raw) ?? raw
  const first = Math.ceil(lo / step - 1e-9) * step
  const out: number[] = []
  for (let v = first; v <= hi + 1e-9; v += step) out.push(Math.round(v / step) * step)
  return out
}

/** A number as a short label. */
const label = (v: number): string => (Math.abs(v) < 1e-9 ? '0' : String(Number(v.toPrecision(3))))

/**
 * One frame of the animation, as a chart: the axes with grid lines, tick labels and axis labels, a legend, the data,
 * the truth, the classic GAM, the weighted bases and their sum, the knots, and beside it the training loss up to this
 * step. It is drawn in shapes (the rasteriser draws no text, so the labels are built from a pixel font).
 */
function frame(r: Result, i: number): Scene {
  const W = 820
  const H = 400
  const edge = mix(MUTED, PAPER, 0.3)
  const grid = 'grid'
  const ink = 'fg'
  const [mx, my, mw, mh] = [58, 70, 470, 282] as const
  const [lx, ly, lw, lh] = [584, 70, 214, 282] as const
  const xLo = r.grid[0] ?? 0
  const xHi = r.grid.at(-1) ?? 1
  const yLo = Math.min(...r.y, ...r.truth) - 0.12
  const yHi = Math.max(...r.y, ...r.truth) + 0.12
  const xs = (v: number): number => mx + ((v - xLo) / (xHi - xLo)) * mw
  const ys = (v: number): number => my + mh - ((Math.max(yLo, Math.min(yHi, v)) - yLo) / (yHi - yLo)) * mh
  const curve = (values: readonly number[]): [number, number][] => values.map((v, k) => [xs(r.grid[k] ?? 0), ys(v)])
  const nodes: SceneNode[] = []
  nodes.push(rect(0, 0, W, H, { fill: 'bg' }))
  // the main plot: grid, ticks and labels
  nodes.push(rect(mx, my, mw, mh, { fill: 'bg', stroke: edge }))
  for (const v of ticks(xLo, xHi, 5)) {
    nodes.push(line(xs(v), my, xs(v), my + mh, { stroke: grid }))
    nodes.push(...pixelText(label(v), xs(v), my + mh + 6, { anchor: 'middle', fill: ink }))
  }
  for (const v of ticks(yLo, yHi, 6)) {
    nodes.push(line(mx, ys(v), mx + mw, ys(v), { stroke: grid }))
    nodes.push(...pixelText(label(v), mx - 6, ys(v) - 3, { anchor: 'end', fill: ink }))
  }
  nodes.push(...pixelText('X', mx + mw / 2, my + mh + 24, { anchor: 'middle', fill: ink }))
  nodes.push(...pixelText('EFFECT', mx, my - 14, { fill: ink }))
  for (const basis of r.bases[i] ?? []) nodes.push(path(curve(basis), { stroke: BASIS, strokeWidth: 1.2 }))
  nodes.push(path(curve(r.truth), { stroke: TRUTH, strokeWidth: 2, dash: [6, 4] }))
  nodes.push(path(curve(r.classic), { stroke: CLASSIC, strokeWidth: 1.6, dash: [2, 3] }))
  nodes.push(path(curve(r.curves[i] ?? []), { stroke: FIT, strokeWidth: 3 }))
  for (const [k, v] of r.x.entries()) nodes.push(circle(xs(v), ys(r.y[k] ?? 0), 2.2, { fill: POINT, opacity: 0.6 }))
  for (const knot of r.knots[i] ?? []) nodes.push(circle(xs(knot), ys(yLo + 0.06), 3.5, { fill: FIT }))
  // the loss so far, on the axes of the whole run, with a mark at this step
  const logged = r.loss.map((v) => Math.log10(v))
  const top = Math.ceil(Math.max(...logged))
  const bottom = Math.floor(Math.min(...logged))
  const lyOf = (v: number): number => ly + lh - ((v - bottom) / (top - bottom)) * lh
  const lxOf = (step: number): number => lx + (step / STEPS) * lw
  nodes.push(rect(lx, ly, lw, lh, { fill: 'bg', stroke: edge }))
  for (let d = bottom; d <= top; d++) {
    nodes.push(line(lx, lyOf(d), lx + lw, lyOf(d), { stroke: grid }))
    nodes.push(...pixelText(label(10 ** d), lx - 6, lyOf(d) - 3, { anchor: 'end', fill: ink }))
  }
  for (const v of ticks(0, STEPS, 4)) {
    nodes.push(line(lxOf(v), ly, lxOf(v), ly + lh, { stroke: grid }))
    nodes.push(...pixelText(label(v), lxOf(v), ly + lh + 6, { anchor: 'middle', fill: ink }))
  }
  nodes.push(...pixelText('STEP', lx + lw / 2, ly + lh + 24, { anchor: 'middle', fill: ink }))
  nodes.push(...pixelText('LOSS', lx, ly - 14, { fill: ink }))
  const upTo = Math.min(r.loss.length, (r.steps[i] ?? 0) + 1)
  const history = logged.slice(0, upTo).map((v, k): [number, number] => [lxOf(k), lyOf(v)])
  if (history.length > 1) nodes.push(path(history, { stroke: 'tab0', strokeWidth: 2 }))
  nodes.push(line(lxOf(r.steps[i] ?? 0), ly, lxOf(r.steps[i] ?? 0), ly + lh, { stroke: ink, dash: [3, 3] }))
  // the title and the legend
  nodes.push(...pixelText(`AFTER ${r.steps[i] ?? 0} STEPS`, 10, 8, { scale: 2, fill: ink }))
  let at = 10
  const entry = (draw: (x: number) => SceneNode[], text: string): void => {
    nodes.push(...draw(at))
    nodes.push(...pixelText(text, at + 24, 40, { fill: ink }))
    at += 24 + pixelTextWidth(text) + 16
  }
  entry((x) => [circle(x + 8, 43, 3, { fill: POINT, opacity: 0.6 })], 'DATA')
  entry((x) => [line(x, 43, x + 18, 43, { stroke: TRUTH, strokeWidth: 2, dash: [5, 3] })], 'TRUTH')
  entry((x) => [line(x, 43, x + 18, 43, { stroke: BASIS, strokeWidth: 2 })], 'WEIGHTED BASIS')
  entry((x) => [line(x, 43, x + 18, 43, { stroke: CLASSIC, strokeWidth: 2, dash: [2, 3] })], 'CLASSIC GAM')
  entry((x) => [line(x, 43, x + 18, 43, { stroke: FIT, strokeWidth: 3 })], 'SPLINE, NO PENALTY')
  entry((x) => [circle(x + 8, 43, 3.5, { fill: FIT })], 'KNOTS')
  return scene(W, H, nodes)
}

export function View({ result }: { result: Result }) {
  const last = result.steps.length - 1
  const yLo = Math.min(...result.y, ...result.truth) - 0.12
  const yHi = Math.max(...result.y, ...result.truth) + 0.12
  const lo = result.grid[0] ?? 0
  const hi = result.grid.at(-1) ?? 1
  const main = cartesian(
    [
      points([{ x: [...result.x], y: [...result.y], name: 'data', colour: POINT, opacity: 0.55 }]),
      lines([{ x: [...result.grid], y: [...result.truth], name: 'truth', colour: TRUTH, width: 2, dash: 'dashed' }]),
      ...(result.bases[last] ?? []).map((basis, k) =>
        lines([
          {
            x: [...result.grid],
            y: [...basis],
            name: k === 0 ? 'weighted basis' : '',
            colour: FIT,
            width: 1.2,
            dash: 'dotted',
            opacity: 0.55,
          },
        ]),
      ),
      lines([
        {
          x: [...result.grid],
          y: [...result.classic],
          name: 'classic GAM (REML)',
          colour: CLASSIC,
          width: 1.6,
          dash: 'dotted',
        },
      ]),
      lines([
        { x: [...result.grid], y: [...(result.curves[last] ?? [])], name: 'spline, no penalty', colour: FIT, width: 3 },
      ]),
      points([
        {
          x: [...(result.knots[last] ?? [])],
          y: (result.knots[last] ?? []).map(() => yLo + 0.05),
          name: 'knots',
          colour: FIT,
        },
      ]),
    ],
    {
      x: { label: 'x', limits: [lo, hi] },
      y: { label: 'effect', limits: [yLo, yHi] },
      legend: true,
    },
  )
  const loss = cartesian([lines([{ y: [...result.loss], name: 'training loss', colour: 'tab0', width: 2 }])], {
    x: { label: 'step' },
    y: { label: 'loss', scale: 'log' },
    legend: false,
  })
  return (
    <Figure width={1040} height={430} title={meta.alt}>
      <Panel x={0} width={710} height={420} still={main} />
      <Panel x={720} width={320} height={420} still={loss} />
    </Figure>
  )
}
