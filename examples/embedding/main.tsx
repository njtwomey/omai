/**
 * Example "A rolled-up sheet, flattened" (in the index, not in the README): a Swiss roll is a flat sheet rolled up in
 * three dimensions, and the generator keeps each point's position along the sheet. Three panels of the same 200 points
 * on one colour scale (the position along the sheet): the data seen along the roll's axis (x against z), where the
 * sheet shows as a spiral; its PCA projection, which can only look at the roll from one side, so layers land on top
 * of each other; and the t-SNE embedding, fitted as a session and read back at its last step. Under them, the objective
 * t-SNE minimised, read back from the same session. The printed numbers measure the layout.
 */
import { generate } from 'mdml-engine/data'
import { swissRoll } from 'mdml-engine/datasets'
import { pca } from 'mdml-engine/decomposition'
import { tsne } from 'mdml-engine/manifold'
import { key } from 'mdml-engine/random'
import { toArray, toFlat } from 'mdml-engine/tensor'
import { session } from 'mdml-engine/trace'
import { cartesian, lines, points } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { colourScale, sequentialRamp } from 'mdml-render/theme'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'A rolled-up sheet, flattened',
  summary: 'A Swiss roll as it is, by PCA and by t-SNE, each point coloured by its true position along the sheet.',
  readme: false,
  alt: 'Three panels of a Swiss roll coloured by position along the sheet: the roll seen end-on as a spiral, its PCA projection where layers overlap, and its t-SNE embedding, with the t-SNE objective under them',
}

interface Result {
  /** The data's x and z coordinates, the PCA scores and the t-SNE embedding, one `[a, b]` per point. */
  readonly original: number[][]
  readonly linear: number[][]
  readonly nonlinear: number[][]
  /** Each point's true position along the sheet (the angle t). */
  readonly along: number[]
  /** The t-SNE objective (KL divergence) at the steps it was read back. */
  readonly steps: number[]
  readonly objective: number[]
}

const N = 200
const STEPS = 300

/** The length along the roll's spiral from the centre to angle t: the spiral `(t cos t, t sin t)` has speed `sqrt(1 + t²)`. */
const arc = (t: number): number => 0.5 * (t * Math.sqrt(t * t + 1) + Math.asinh(t))

/**
 * How far apart on the sheet the 5 nearest neighbours of a point (in `layout`, in any number of dimensions) are, on average. The sheet distance
 * is measured in the sheet's own coordinates (length along the spiral, height across it), in the data's units.
 */
function neighbourGap(layout: readonly (readonly number[])[], sheet: readonly (readonly [number, number])[]): number {
  const apart = (a: readonly [number, number], b: readonly [number, number]): number =>
    Math.hypot(a[0] - b[0], a[1] - b[1])
  let total = 0
  layout.forEach((p, i) => {
    const near = layout
      .map((q, j) => ({ j, d: Math.hypot(...p.map((v, c) => v - (q[c] ?? 0))) }))
      .filter((e) => e.j !== i)
      .sort((a, b) => a.d - b.d)
      .slice(0, 5)
    total += near.reduce((s, e) => s + apart(sheet[e.j] as [number, number], sheet[i] as [number, number]), 0) / 5
  })
  return total / layout.length
}

export function run(): Outcome<Result> {
  const { data, truth } = generate(swissRoll, { knobs: { n: N }, key: key(0) })
  const along = Array.from(toFlat(truth.latents?.['t'] as never)) // the true position along the sheet
  const height = Array.from(toFlat(truth.latents?.['height'] as never))
  const sheet = along.map((t, i): [number, number] => [arc(t), height[i] ?? 0])
  const space = toArray(data.x) as number[][]

  const linear = pca.with({ nComponents: 2 }).fit(data)
  const flat = toArray(linear.transform(data.x)) as number[][]

  // The session is made from the method so that a checkpoint is kept every 100 steps (the tape stays small).
  const method = tsne.with({ perplexity: 30, maxIter: STEPS, exaggerationIter: 100 }).method(data)
  const fit = session(method, { key: key(1), steps: STEPS, checkpointEvery: 100 })
  fit.advance()
  const steps = Array.from({ length: STEPS / 10 + 1 }, (_, i) => i * 10)
  const objective = steps.map((t) => (fit.probe(t, 'model/params') as { objective: number }).objective)
  const embedded = fit.probe(fit.t, 'model/embedding') as number[][]

  const pairs = sheet.flatMap((a, i) => sheet.slice(i + 1).map((b) => Math.hypot(a[0] - b[0], a[1] - b[1])))
  const notes = [
    `${fit.t} t-SNE steps, ${fit.status}; KL divergence ${objective[0]?.toFixed(3)} at step 0, ${objective[10]?.toFixed(3)} at step 100, ${objective.at(-1)?.toFixed(3)} at the end`,
    `mean sheet distance (length along the sheet and height, in the data's units) between a point and its 5 nearest neighbours: in 3-D space ${neighbourGap(space, sheet).toFixed(2)}, PCA plane ${neighbourGap(flat, sheet).toFixed(2)}, t-SNE plane ${neighbourGap(embedded, sheet).toFixed(2)}; two random points are ${(pairs.reduce((a, b) => a + b, 0) / pairs.length).toFixed(2)} apart`,
    `the sheet spans angle t from ${Math.min(...along).toFixed(1)} to ${Math.max(...along).toFixed(1)}, ${arc(Math.max(...along)).toFixed(1) } minus ${arc(Math.min(...along)).toFixed(1)} long`,
  ]
  return {
    result: {
      original: space.map((r) => [r[0] ?? 0, r[2] ?? 0]),
      linear: flat,
      nonlinear: embedded,
      along,
      steps,
      objective,
    },
    tapes: { session: fit },
    notes,
  }
}

const FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
const PANEL = { width: 370, height: 330 }
const BAR = 120
const PAD = { top: 14, right: 12, bottom: 44, left: 52 }

export function View({ result }: { result: Result }) {
  const lo = Math.min(...result.along)
  const hi = Math.max(...result.along)
  const scatter = (rows: readonly (readonly number[])[], xLabel: string, yLabel: string): unknown =>
    cartesian(
      [
        points([
          {
            x: rows.map((r) => r[0] ?? 0),
            y: rows.map((r) => r[1] ?? 0),
            values: result.along,
            size: 8,
            opacity: 1,
            name: 'points',
          },
        ]),
      ],
      {
        figure: figure({ ...PANEL, padding: PAD }),
        x: { label: xLabel },
        y: { label: yLabel },
        legend: false,
      },
    )
  const head = 36
  const top = 26 // within a panel's group, below its title
  const curveTop = head + top + PANEL.height + 6
  const curve = cartesian(
    [lines([{ x: result.steps, y: result.objective, colour: 'tab0', width: 2.5, name: 'KL' }])],
    {
      figure: figure({
        width: 3 * PANEL.width,
        height: 170,
        padding: { top: 10, right: 12, bottom: 40, left: 52 },
      }),
      x: { label: 'step' },
      y: { label: 'KL divergence', limits: [0, 2] },
      legend: false,
    },
  )
  // One colour bar for all three panels: viridis along the sheet, the same ramp the points use.
  const ramp = sequentialRamp('viridis')
  const at = colourScale(ramp, [lo, hi])
  const barTop = head + top + PAD.top
  const barHeight = PANEL.height - PAD.top - PAD.bottom
  const barX = 3 * PANEL.width + 24
  const ticks = [Math.ceil(lo), Math.floor((lo + hi) / 2), Math.floor(hi)]
  return (
    <Figure width={3 * PANEL.width + BAR} height={curveTop + 26 + 170 + 58} title={meta.alt}>
      <text x={(3 * PANEL.width + BAR) / 2} y={24} textAnchor="middle" fontFamily={FONT} fontSize={19} fontWeight={700} fill="#222222">
        A Swiss roll is a 2-D sheet curled up in 3-D: PCA flattens it badly, t-SNE keeps neighbours together
      </text>
      <g transform={`translate(0 ${head})`}>
      <Panel x={0} width={PANEL.width} height={PANEL.height} title="the data: x against z" still={scatter(result.original, 'x', 'z')} />
      <Panel
        x={PANEL.width}
        width={PANEL.width}
        height={PANEL.height}
        title="PCA, two components"
        still={scatter(result.linear, 'component 1', 'component 2')}
      />
      <Panel
        x={2 * PANEL.width}
        width={PANEL.width}
        height={PANEL.height}
        title="t-SNE, 300 steps"
        still={scatter(result.nonlinear, 't-SNE 1', 't-SNE 2')}
      />
      </g>
      <defs>
        <linearGradient id="along" x1="0" y1="1" x2="0" y2="0">
          {Array.from({ length: 11 }, (_, i) => (
            <stop key={i} offset={`${i * 10}%`} stopColor={at(lo + ((hi - lo) * i) / 10)} />
          ))}
        </linearGradient>
      </defs>
      <rect x={barX} y={barTop} width={16} height={barHeight} fill="url(#along)" />
      {ticks.map((t) => {
        const y = barTop + barHeight * (1 - (t - lo) / (hi - lo))
        return (
          <g key={t} fontFamily={FONT} fontSize={12} fill="#444444">
            <line x1={barX + 16} x2={barX + 21} y1={y} y2={y} stroke="#444444" />
            <text x={barX + 25} y={y + 4}>
              {t}
            </text>
          </g>
        )
      })}
      <text
        transform={`translate(${barX + 62} ${barTop + barHeight / 2}) rotate(90)`}
        textAnchor="middle"
        fontFamily={FONT}
        fontSize={12}
        fill="#444444"
      >
        position along the sheet (angle t)
      </text>
      <g transform={`translate(0 ${curveTop})`}>
        <Panel width={3 * PANEL.width} height={170} title="what t-SNE minimised: KL divergence against step" still={curve} />
      </g>
      <text x={8} y={curveTop + 26 + 170 + 26} fontFamily={FONT} fontSize={14} fill="#333333">
        Colour is each point's true position along the sheet, so a good layout is a smooth colour gradient. PCA overlaps layers of the roll; t-SNE keeps
      </text>
      <text x={8} y={curveTop + 26 + 170 + 46} fontFamily={FONT} fontSize={14} fill="#333333">
        neighbours together but cuts the sheet into pieces rather than unrolling it.
      </text>
    </Figure>
  )
}
