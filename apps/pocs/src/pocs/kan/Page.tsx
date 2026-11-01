import { generate, type RegressionData } from 'mdml-engine/data'
import { kanFunctions } from 'mdml-engine/datasets'
import { kanRegressor } from 'mdml-engine/learn'
import { kanBases } from 'mdml-engine/nn'
import { key } from 'mdml-engine/random'
import { choice, defaults, dict, int, real } from 'mdml-engine/space'
import { asTensor, toFlat } from 'mdml-engine/tensor'
import type { Config } from 'mdml-engine/trace'
import { metricSeries } from 'mdml-engine/trace'
import { cartesian, lines } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import {
  Chart,
  ControlPanel,
  type LiveSession,
  SessionPlayer,
  SpaceControls,
  useColourMode,
  useProbe,
  useRun,
  useScrubClick,
  useSession,
  useSessionPreferences,
} from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'

const LOSS = figure({ width: 420, height: 300 })

const SPACE = dict({
  function: choice(['exp-sin-square', 'multiplication'], {
    default: 'exp-sin-square',
    label: 'the function to learn',
    descriptions: {
      'exp-sin-square':
        'exp(sin(π x₀) + x₁²): a sine and a square feed one node, then an exponential (a [2, 1, 1] network suffices)',
      multiplication: 'x₀ x₁ = ((x₀ + x₁)² − (x₀ − x₁)²) / 4: needs a hidden layer of two nodes ([2, 2, 1])',
    },
  }),
  noise: real(0, 0.5, { default: 0, label: 'noise standard deviation' }),
  hidden: int(1, 8, { default: 3, label: 'hidden nodes' }),
  grid: int(2, 20, { default: 5, label: 'grid size (intervals of each edge curve)' }),
  degree: int(0, 5, { default: 3, label: 'spline degree (0 constant, 1 linear, 3 cubic)' }),
  basis: choice(['bspline', 'rbf', 'chebyshev', 'fourier'], {
    default: 'bspline',
    label: 'family of edge curves',
    descriptions: {
      bspline: 'B-splines on a grid of knots (the paper)',
      rbf: 'Gaussian radial basis functions on a grid of centres (FastKAN)',
      chebyshev: 'Chebyshev polynomials (ChebyKAN): global, no grid',
      fourier: 'sines and cosines of increasing frequency: global, no grid',
    },
  }),
  lambdaL1: real(0, 0.2, { default: 0, label: 'sparsity: L1 penalty on the edges' }),
  lambdaEntropy: real(0, 2, { default: 0, label: 'sparsity: entropy penalty per layer' }),
  epochs: int(20, 600, { default: 200, log: true, label: 'steps' }),
  learningRate: real(0.002, 0.2, { default: 0.03, log: true, label: 'learning rate' }),
})
type Controls = {
  function: string
  noise: number
  hidden: number
  grid: number
  degree: number
  basis: string
  lambdaL1: number
  lambdaEntropy: number
  epochs: number
  learningRate: number
}
const START: Controls = {
  function: 'exp-sin-square',
  noise: 0,
  hidden: 3,
  grid: 5,
  degree: 3,
  basis: 'bspline',
  lambdaL1: 0,
  lambdaEntropy: 0,
  epochs: 200,
  learningRate: 0.03,
}
const ROWS = 300

export default function KanPage(): ReactNode {
  const [controls, setControls] = useState<Controls>(START)
  return (
    <div className="flex flex-col gap-4">
      <ControlPanel title="the function, the network and its penalties" columns={2}>
        <SpaceControls space={SPACE} value={controls} onChange={(v) => setControls(v as unknown as Controls)} />
      </ControlPanel>
      <Training controls={controls} reset={() => setControls(START)} />
    </div>
  )
}

function Training(props: { readonly controls: Controls; readonly reset: () => void }): ReactNode {
  const { controls, reset } = props
  const prefs = useSessionPreferences()
  const [at, setAt] = useState<number | null>(null)
  const data = useMemo(() => {
    const g = generate(kanFunctions, {
      key: key(0),
      knobs: { n: ROWS, function: controls.function, noise: controls.noise },
    })
    const table = g.data as RegressionData
    const x = table.x
    const d = x.shape[1] ?? 2
    const flat = toFlat(asTensor(x))
    return {
      x: Array.from({ length: ROWS }, (_, i) => Array.from(flat.slice(i * d, (i + 1) * d))),
      y: Array.from(toFlat(asTensor(table.y))),
    }
  }, [controls.function, controls.noise])
  const method = useMemo(() => {
    const family = { ...(defaults(kanBases.space.options[controls.basis] as never) as object), kind: controls.basis }
    const model = kanRegressor.with({
      hidden: [controls.hidden],
      grid: controls.grid,
      degree: controls.degree,
      basis: family as never,
      lambdaL1: controls.lambdaL1,
      lambdaEntropy: controls.lambdaEntropy,
      epochs: controls.epochs,
      learningRate: controls.learningRate,
      curvePoints: 41,
    })
    return model.method(data)
  }, [data, controls])
  const live = useSession(
    method,
    { config: defaults(method.config) as Config, mode: prefs.mode, steps: controls.epochs },
    [method],
    { hold: prefs.hold },
  )
  return (
    <div className="flex flex-col gap-3">
      <View live={live} at={at} onScrub={setAt} />
      <SessionPlayer live={live} at={at} onScrub={setAt} onReset={reset} />
      <p className="max-w-3xl text-xs text-muted">
        Each edge of the network carries its own curve; the picture shows every edge's curve over its input's range,
        drawn as strongly as the edge matters (its mean magnitude on the data). An edge that matters little is nearly
        invisible: that is what the sparsity penalties push towards. Scrub the loss to see the curves form.
      </p>
    </div>
  )
}

interface Edge {
  readonly layer: number
  readonly i: number
  readonly j: number
  readonly grid: number[]
  readonly values: number[]
  readonly scale: number
  readonly active: boolean
}

const W = 900
const H = 400
const PAD = 70
const BOX = { w: 78, h: 54 }

function View(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the session is generic over the method
  readonly live: LiveSession<any, any>
  readonly at: number | null
  readonly onScrub: (t: number | null) => void
}): ReactNode {
  const { live } = props
  const mode = useColourMode()
  const run = useRun(live)
  const t = Math.max(0, Math.min(props.at ?? live.head, live.head))
  const curves = useProbe(live, t, 'kan/curves')
  const sparsity = useProbe(live, t, 'kan/sparsity')
  const loss = useMemo(() => metricSeries(run, 'train/loss'), [run])
  const steps = useMemo(() => Array.from({ length: live.head + 1 }, (_, i) => i), [live.head])
  const seek = useScrubClick(steps, live.head, props.onScrub)
  const value = curves.value as { widths: number[]; edges: Edge[] } | undefined
  const lossOption = useMemo(() => {
    const upto = loss.steps.map((st, i) => [st, loss.values[i] ?? Number.NaN] as const).filter(([st]) => st <= t)
    return cartesian(
      [
        lines([{ x: upto.map((p) => p[0]), y: upto.map((p) => p[1]), name: 'training loss', width: 2 }], { mode }),
        lines(
          [
            {
              x: [t, t],
              y: [Math.min(...loss.values), Math.max(...loss.values)],
              colour: 'muted',
              dash: 'dashed',
              width: 1,
            },
          ],
          { mode },
        ),
      ],
      {
        mode,
        figure: LOSS,
        x: { label: 'step', limits: [0, Math.max(1, live.head)] },
        y: { label: 'loss', scale: 'log' },
      },
    )
  }, [loss, t, mode, live.head])
  const share = typeof sparsity.value === 'number' ? sparsity.value : undefined
  return (
    <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[minmax(0,2.4fr)_minmax(0,1fr)]">
      {value === undefined ? (
        <p className="text-sm text-muted">{curves.error ?? 'Press play to train.'}</p>
      ) : (
        <Network widths={value.widths} edges={value.edges} />
      )}
      <div className="flex flex-col gap-2">
        <Chart option={lossOption} figure={LOSS} onInit={seek} />
        <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
          <div>
            <dt className="text-xs text-muted">step</dt>
            <dd className="font-medium tabular-nums">
              {t} of {live.head}
            </dd>
          </div>
          <div>
            <dt className="text-xs text-muted">negligible edges</dt>
            <dd className="font-medium tabular-nums">{share === undefined ? '' : `${Math.round(100 * share)}%`}</dd>
          </div>
        </dl>
      </div>
    </div>
  )
}

/** The network as nodes in columns, each edge with its curve in a small box on it, drawn as strongly as it matters. */
function Network(props: { readonly widths: readonly number[]; readonly edges: readonly Edge[] }): ReactNode {
  const { widths, edges } = props
  const layers = widths.length
  const x = (l: number): number => PAD + (l * (W - 2 * PAD)) / Math.max(1, layers - 1)
  const y = (l: number, n: number): number => {
    const count = widths[l] ?? 1
    return H / 2 + (n - (count - 1) / 2) * Math.min(90, (H - 80) / Math.max(1, count))
  }
  const biggest = Math.max(1e-9, ...edges.map((e) => e.scale))
  // The curves of a layer share one vertical scale, so a small edge looks small.
  const peak = (l: number): number =>
    Math.max(1e-9, ...edges.filter((e) => e.layer === l && e.active).flatMap((e) => e.values.map((v) => Math.abs(v))))
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full rounded-lg border border-line bg-bg"
      role="img"
      aria-label="the network"
    >
      {edges.map((e) => {
        const [x0, y0] = [x(e.layer), y(e.layer, e.i)]
        const [x1, y1] = [x(e.layer + 1), y(e.layer + 1, e.j)]
        const strength = e.active ? Math.min(1, Math.sqrt(e.scale / biggest)) : 0
        // The box sits a third of the way along the edge, so the boxes of the edges into one node do not collide.
        const bx = x0 + (x1 - x0) * 0.4
        const by = y0 + (y1 - y0) * 0.4
        const p = peak(e.layer)
        const lo = e.grid[0] ?? 0
        const hi = e.grid[e.grid.length - 1] ?? 1
        const path = e.values
          .map((v, k) => {
            const px = bx - BOX.w / 2 + (((e.grid[k] ?? lo) - lo) / (hi - lo || 1)) * BOX.w
            const py = by - (v / p) * (BOX.h / 2 - 4)
            return `${k === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`
          })
          .join('')
        return (
          <g key={`${e.layer}-${e.i}-${e.j}`} opacity={0.15 + 0.85 * strength}>
            <line x1={x0} y1={y0} x2={x1} y2={y1} stroke="var(--wb-grid)" strokeWidth={0.5 + 3 * strength} />
            <rect
              x={bx - BOX.w / 2}
              y={by - BOX.h / 2}
              width={BOX.w}
              height={BOX.h}
              rx={4}
              fill="var(--wb-bg)"
              stroke="var(--wb-grid)"
            />
            <path d={path} fill="none" stroke="var(--wb-accent)" strokeWidth={1.6} />
          </g>
        )
      })}
      {widths.map((count, l) =>
        Array.from({ length: count }, (_, n) => (
          <circle
            // biome-ignore lint/suspicious/noArrayIndexKey: nodes are positional
            key={`${l}-${n}`}
            cx={x(l)}
            cy={y(l, n)}
            r={9}
            fill="var(--wb-surface)"
            stroke="var(--wb-fg)"
            strokeWidth={1.5}
          />
        )),
      )}
      {widths.map((count, l) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: layers are positional
        <text key={`t${l}`} x={x(l)} y={H - 10} textAnchor="middle" fontSize={12} fill="var(--wb-muted)">
          {l === 0 ? `inputs (${count})` : l === layers - 1 ? `output (${count})` : `hidden (${count})`}
        </text>
      ))}
    </svg>
  )
}
