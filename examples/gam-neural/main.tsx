/**
 * Example "A neural additive model, learning its shapes": `y` is the sum of three smooth functions, one of each of
 * three features, plus noise. The model has one small network per feature, summed (a neural generalised additive
 * model), so what it learns is three curves, and the data generator knows the true ones. The training is a session:
 * its tape holds the loss at every step, and asking it for each network's curve at a step gives the frames of the
 * animation `learning.gif`, where the fitted curves move toward the true curves while the loss falls.
 */
import { generate } from 'mdml-engine/data'
import { additiveModel } from 'mdml-engine/datasets'
import { additiveRegressor } from 'mdml-engine/gam'
import { netTerm } from 'mdml-engine/nn'
import { key } from 'mdml-engine/random'
import { line, path, rect, type Scene, scene, timeline } from 'mdml-engine/scene'
import { type Tensor, tensor, toFlat } from 'mdml-engine/tensor'
import { Log, metricSeries } from 'mdml-engine/trace'
import { cartesian, lines } from 'mdml-render/charts'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Row } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'A neural additive model, learning its shapes',
  summary:
    'One small network per feature, summed: the three fitted curves move toward the true ones as the loss falls (learning.gif).',
  readme: false,
  alt: 'The three shape functions of a neural additive model against the true ones after training, and its training loss',
}

const STEPS = 250

/** What the figure and the animation are drawn from: each feature's grid, true curve and curves at the steps shown. */
interface Result {
  readonly grids: readonly (readonly number[])[]
  readonly truth: readonly (readonly number[])[]
  /** `fits[f][i]`: feature f's curve after `steps[i]` steps. */
  readonly fits: readonly (readonly (readonly number[])[])[]
  readonly steps: readonly number[]
  readonly loss: readonly number[]
  readonly names: readonly string[]
}

export function run(): Outcome<Result> {
  const { data, truth } = generate(additiveModel, {
    knobs: { n: 200, nFeatures: 3, nInformative: 3, noise: 0.3 },
    key: key(0),
  })
  const model = additiveRegressor.with({
    terms: [0, 1, 2].map((feature) => netTerm(feature, { width: 12, depth: 2 })),
    epochs: STEPS,
    learningRate: 0.02,
    centreRows: 16,
  })
  // A step budget much longer than the training keeps the session's checkpoints few, so the tape stays small; asking
  // for an earlier step replays the training from the nearest checkpoint.
  const training = model.session(data, { key: key(1), steps: 20000, logs: { 'train/loss': Log() } })
  training.advance()
  const steps = [0, 1, 2, 4, 7, 11, 16, 23, 32, 45, 60, 80, 105, 135, 170, 210, STEPS]
  const probes = steps.map((t) =>
    [0, 1, 2].map((f) => training.probe(t, `predictor/0/term/${f}/curve`) as { grid: number[]; values: number[] }),
  )
  // The true curve of each feature, centred the way a network's term is: its mean over the training rows is zero.
  const components = (truth.extra as { components: { name: string; f: (x: Tensor) => Tensor }[] }).components
  const x = data.x
  const grids = [0, 1, 2].map((f) => probes[0]?.[f]?.grid ?? [])
  const centred = grids.map((grid, f) => {
    const component = components[f]?.f as (v: Tensor) => Tensor
    const at = (values: readonly number[]): number[] =>
      toFlat(component(tensor(values.map((v) => [0, 1, 2].map((c) => (c === f ? v : 0))))))
    const rows = Array.from({ length: x.shape[0] ?? 0 }, (_, i) => toFlat(x)[i * 3 + f] ?? 0)
    const mean = at(rows).reduce((a, b) => a + b, 0) / rows.length
    return at(grid).map((v) => v - mean)
  })
  const result: Result = {
    grids,
    truth: centred,
    fits: [0, 1, 2].map((f) => probes.map((p) => p[f]?.values ?? [])),
    steps,
    loss: metricSeries(training.run(), 'train/loss').values,
    names: components.slice(0, 3).map((c) => c.name),
  }
  const last = result.fits.map((f, k) => rmse(f.at(-1) ?? [], centred[k] ?? []))
  const notes = [
    `${STEPS} steps; training loss ${result.loss[0]?.toFixed(2)} at the start, ${result.loss.at(-1)?.toFixed(3)} at the end (noise variance 0.09)`,
    `root mean square gap between each fitted curve and the true one: ${last.map((v) => v.toFixed(3)).join(', ')}`,
    `${result.steps.length} frames of the animation at steps ${result.steps.join(' ')}`,
  ]
  const shown = result.steps.map((step, i) => ({
    kind: 'scene' as const,
    step,
    scene: frame(result, i),
    caption: `step ${step}`,
  }))
  const hold = shown.slice(-1) // the last frame is shown three more times, so the loop pauses on the result
  const animation = timeline([...shown, ...hold, ...hold, ...hold], { fps: 6 })
  return { result, tapes: { session: training }, animation, notes }
}

const rmse = (a: readonly number[], b: readonly number[]): number =>
  Math.sqrt(a.reduce((s, v, i) => s + (v - (b[i] ?? 0)) ** 2, 0) / Math.max(1, a.length))

const TRUE = '#6b7280'
const FIT = '#1f77b4'
const EDGE = '#cbd5e1'

/**
 * One frame of the animation as a scene, in shapes only (the rasteriser draws no text): three panels, one per feature,
 * each with the true curve in grey and the fitted curve in blue; under them the loss curve up to this step, and a bar
 * that fills as training goes.
 */
function frame(r: Result, i: number): Scene {
  const W = 540
  const H = 262
  const all = [...r.truth.flat(), ...r.fits.flatMap((f) => f.flat())]
  const lo = Math.min(...all)
  const hi = Math.max(...all)
  const nodes = []
  r.grids.forEach((grid, f) => {
    const [x0, y0, w, h] = [10 + f * 175, 10, 165, 150] as const
    const px = (v: number): number => x0 + ((v - (grid[0] ?? 0)) / ((grid.at(-1) ?? 1) - (grid[0] ?? 0))) * w
    const py = (v: number): number => y0 + h - ((v - lo) / (hi - lo)) * h
    const curve = (values: readonly number[]): [number, number][] => values.map((v, k) => [px(grid[k] ?? 0), py(v)])
    nodes.push(rect(x0, y0, w, h, { fill: '#ffffff', stroke: EDGE }))
    nodes.push(line(x0, py(0), x0 + w, py(0), { stroke: EDGE, dash: [3, 3] }))
    nodes.push(path(curve(r.truth[f] ?? []), { stroke: TRUE, strokeWidth: 2.5 }))
    nodes.push(path(curve(r.fits[f]?.[i] ?? []), { stroke: FIT, strokeWidth: 2 }))
  })
  // The loss so far, on the axes of the whole run.
  const [lx, ly, lw, lh] = [10, 172, 520, 70] as const
  const logged = r.loss.map((v) => Math.log10(v)) // on a log axis, as the loss falls by more than an order of magnitude
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
  nodes.push(rect(lx, 250, lw, 4, { fill: EDGE }))
  nodes.push(rect(lx, 250, (lw * (r.steps[i] ?? 0)) / STEPS, 4, { fill: FIT }))
  return scene(W, H, nodes)
}

export function View({ result }: { result: Result }) {
  const last = result.steps.length - 1
  const panel = (f: number) => ({
    title: result.names[f] ?? `x${f}`,
    still: cartesian(
      [
        lines([{ x: result.grids[f] ?? [], y: result.truth[f] ?? [], name: 'true curve', colour: TRUE, width: 2.5 }]),
        lines([{ x: result.grids[f] ?? [], y: result.fits[f]?.[last] ?? [], name: 'learned', colour: FIT, width: 2 }]),
      ],
      { x: { label: `x${f}` }, y: { label: f === 0 ? 'contribution to y' : '' }, legend: f === 0 },
    ),
  })
  return (
    <Row
      alt={meta.alt}
      width={330}
      height={300}
      panels={[
        panel(0),
        panel(1),
        panel(2),
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
