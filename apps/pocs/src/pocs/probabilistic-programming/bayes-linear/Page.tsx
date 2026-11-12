import { generate, type RegressionData } from 'omai-engine/data'
import { lineData } from 'omai-engine/datasets'
import { bayesianOnlineLinear } from 'omai-engine/linear'
import { key } from 'omai-engine/random'
import { defaults, dict, int, real } from 'omai-engine/space'
import { asTensor, toFlat } from 'omai-engine/tensor'
import type { Config } from 'omai-engine/trace'
import { metricSeries } from 'omai-engine/trace'
import { band, cartesian, heatmap, lines, points } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import {
  Chart,
  ControlPanel,
  type LiveSession,
  SessionPlayer,
  SpaceControls,
  useColourMode,
  useProbe,
  useRun,
  useSession,
  useSessionPreferences,
} from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { StaleVeil } from '../../../shared/StaleVeil.tsx'

// The picture is laid out aaa / aaa / aaa / bcd on three columns: the regression is three squares wide and tall, and
// the prior, the likelihood and the posterior are one square each beneath it.
const CELL = 288
const GAP = 16
const MAIN = figure({ width: 3 * CELL + 2 * GAP, height: 3 * CELL + 2 * GAP })
const SQUARE = figure({ width: CELL, height: CELL })
const FG = '#e6edf3'
const ORANGE = '#e8743b'
const GRID = 121

const SPACE = dict({
  intercept: real(-3, 3, { default: 0.5, label: 'true intercept' }),
  slope: real(-3, 3, { default: 1.5, label: 'true slope' }),
  noise: real(0.05, 2, { default: 0.5, log: true, label: 'true noise (standard deviation)' }),
  modelNoise: real(0.05, 2, { default: 0.5, log: true, label: 'noise the model assumes' }),
  priorStd: real(0.1, 5, { default: 1.5, log: true, label: 'prior standard deviation of the weights' }),
  n: int(2, 100, { default: 30, label: 'points' }),
  seed: int(0, 99, { default: 3, label: 'seed' }),
})
type Controls = {
  intercept: number
  slope: number
  noise: number
  modelNoise: number
  priorStd: number
  n: number
  seed: number
}
const START: Controls = { intercept: 0.5, slope: 1.5, noise: 0.5, modelNoise: 0.5, priorStd: 1.5, n: 30, seed: 3 }

export default function BayesLinearPage(): ReactNode {
  const [controls, setControls] = useState<Controls>(START)
  return (
    <div className="flex flex-col gap-4">
      <ControlPanel title="the truth, the noise and the prior" columns={3}>
        <SpaceControls space={SPACE} value={controls} onChange={(v) => setControls(v as unknown as Controls)} />
      </ControlPanel>
      <Run controls={controls} reset={() => setControls(START)} />
    </div>
  )
}

function Run(props: { readonly controls: Controls; readonly reset: () => void }): ReactNode {
  const { controls, reset } = props
  const prefs = useSessionPreferences()
  const [at, setAt] = useState<number | null>(null)
  const data = useMemo(() => {
    const g = generate(lineData, {
      key: key(controls.seed),
      knobs: { n: controls.n, noise: controls.noise, intercept: controls.intercept, slope: controls.slope },
    })
    const d = g.data as RegressionData
    return {
      x: Array.from(toFlat(asTensor(d.x))),
      y: Array.from(toFlat(asTensor(d.y))),
    }
  }, [controls.n, controls.noise, controls.intercept, controls.slope, controls.seed])
  const method = useMemo(
    () =>
      bayesianOnlineLinear
        .with({ alpha: 1 / controls.priorStd ** 2, beta: 1 / controls.modelNoise ** 2, intercept: true })
        .method({ x: data.x.map((v) => [v]), y: data.y }),
    [data, controls.priorStd, controls.modelNoise],
  )
  const live = useSession(
    method,
    { config: defaults(method.config) as Config, mode: prefs.mode, steps: controls.n },
    [method],
    { hold: prefs.hold },
  )
  return (
    <div className="flex flex-col gap-3">
      <StaleVeil stale={live.stale}>
        <View live={live} at={at} data={data} controls={controls} />
      </StaleVeil>
      <SessionPlayer live={live} at={at} onScrub={setAt} onReset={reset} />
      <p className="max-w-3xl text-xs text-muted">
        Each point multiplies the belief about the weights by its likelihood, a ridge in weight space (every pair of
        weights that passes through the point equally well); the product is the posterior, and it becomes the prior for
        the next point. In the squares the white dot marks the true weights and the orange dot the mean of the belief.
        The lines in the top chart are drawn from the posterior.
      </p>
    </div>
  )
}

function View(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the session is generic over the method
  readonly live: LiveSession<any, any>
  readonly at: number | null
  readonly data: { readonly x: number[]; readonly y: number[] }
  readonly controls: Controls
}): ReactNode {
  const { live, data, controls } = props
  const mode = useColourMode()
  const run = useRun(live)
  const t = Math.max(0, Math.min(props.at ?? live.head, live.head))
  const reach = Math.max(
    2.5 * controls.priorStd,
    1.5 * Math.max(Math.abs(controls.intercept), Math.abs(controls.slope)) + 0.5,
  )
  const xs = useMemo(() => Array.from({ length: 41 }, (_, i) => -1 + (2 * i) / 40), [])
  const lo = Math.min(...data.y, controls.intercept - Math.abs(controls.slope)) - 0.6
  const hi = Math.max(...data.y, controls.intercept + Math.abs(controls.slope)) + 0.6
  const fit = useProbe(live, t, 'model/lines', { x: xs, samples: 14 })
  const range = useMemo(() => [-reach, reach, -reach, reach], [reach])
  const weights = useProbe(live, t, 'model/weights', { range, points: GRID })
  const evidence = useMemo(() => metricSeries(run, 'blr/log-evidence'), [run])
  const w = weights.value as
    | {
        w0: number[]
        w1: number[]
        prior: number[][]
        likelihood: number[][] | null
        posterior: number[][]
        posteriorMean: number[]
        priorMean: number[]
        observation: { phi: number[]; y: number } | null
      }
    | undefined
  const f = fit.value as { mean: number[]; std: number[]; samples: number[][] } | undefined
  const truthLine = xs.map((x) => controls.intercept + controls.slope * x)

  const main = useMemo(() => {
    const seenX = data.x.slice(0, t)
    const seenY = data.y.slice(0, t)
    const last = t > 0 ? [data.x[t - 1] ?? 0, data.y[t - 1] ?? 0] : undefined
    return cartesian(
      [
        ...(f === undefined
          ? []
          : [
              band(
                [
                  {
                    x: xs,
                    lower: f.mean.map((m, i) => m - 2 * (f.std[i] ?? 0)),
                    upper: f.mean.map((m, i) => m + 2 * (f.std[i] ?? 0)),
                  },
                ],
                { mode },
              ),
              lines(
                f.samples.map((s) => ({ x: xs, y: s, colour: ORANGE, width: 1, opacity: 0.35 })),
                { mode },
              ),
            ]),
        points(
          [
            { x: data.x.slice(t), y: data.y.slice(t), name: 'to come', opacity: 0.25, colour: 'muted' },
            { x: seenX, y: seenY, name: 'seen', opacity: 0.85 },
            ...(last === undefined
              ? []
              : [{ x: [last[0] as number], y: [last[1] as number], name: 'latest', colour: FG }]),
          ],
          { mode, colourOffset: 7 },
        ),
        lines([{ x: xs, y: truthLine, name: 'true line', dash: 'dashed', width: 2, colour: 'muted' }], { mode }),
        ...(f === undefined
          ? []
          : [lines([{ x: xs, y: f.mean, name: 'posterior mean', width: 3, colour: ORANGE }], { mode })]),
      ],
      {
        mode,
        figure: MAIN,
        title: t === 0 ? 'before any point' : `after ${t} point${t === 1 ? '' : 's'}`,
        x: { label: 'x', limits: [-1, 1] },
        y: { label: 'y', limits: [lo, hi] },
      },
    )
  }, [f, data, t, xs, truthLine, lo, hi, mode])

  const panel = (title: string, z: number[][] | null | undefined, markMean: number[] | undefined, note?: string) => {
    if (w === undefined) return undefined
    const top = z === null || z === undefined ? 1 : Math.max(1e-12, ...z.flat())
    return cartesian(
      [
        ...(z === null || z === undefined
          ? []
          : [heatmap(z, { x: w.w0, y: w.w1, range: [0, top], smooth: 'bilinear', colourBar: false, mode })]),
        points(
          [
            { x: [controls.intercept], y: [controls.slope], name: 'true weights', colour: FG },
            ...(markMean === undefined
              ? []
              : [{ x: [markMean[0] as number], y: [markMean[1] as number], name: 'mean', colour: ORANGE }]),
          ],
          { mode },
        ),
      ],
      {
        mode,
        figure: SQUARE,
        title: note === undefined ? title : `${title} ${note}`,
        legend: false,
        x: { label: 'intercept', limits: [-reach, reach] },
        y: { label: 'slope', limits: [-reach, reach] },
      },
    )
  }
  const prior = panel('prior', w?.prior, w?.priorMean, t === 0 ? '(the prior)' : '(belief before this point)')
  const likelihood = panel(
    'likelihood',
    w?.likelihood,
    undefined,
    w?.likelihood == null ? '(no point yet)' : '(of this point)',
  )
  const posterior = panel('posterior', w?.posterior, w?.posteriorMean, '(prior × likelihood)')

  const sd = (i: number): number => {
    const m = (live.session.stateAt(t) as { covariance: number[][] }).covariance
    return Math.sqrt(m[i]?.[i] ?? Number.NaN)
  }
  const evidenceNow = (() => {
    const i = evidence.steps.findLastIndex((s) => s <= t)
    return i < 0 ? 0 : (evidence.values[i] ?? 0)
  })()
  const facts: [string, string][] = [
    ['step', `${t} of ${controls.n}`],
    [
      'intercept',
      w === undefined ? '' : `${w.posteriorMean[0]?.toFixed(2)} ± ${sd(0).toFixed(2)}  (true ${controls.intercept})`,
    ],
    [
      'slope',
      w === undefined ? '' : `${w.posteriorMean[1]?.toFixed(2)} ± ${sd(1).toFixed(2)}  (true ${controls.slope})`,
    ],
    ['log evidence', evidenceNow.toFixed(2)],
  ]
  return (
    <div className="flex flex-col items-start gap-6 xl:flex-row">
      {/* The layout is in inline styles, so it does not depend on which Tailwind classes the dev server has seen. */}
      <div style={{ display: 'grid', gap: GAP, gridTemplateColumns: `repeat(3, ${CELL}px)`, flex: 'none' }}>
        <div style={{ gridColumn: '1 / span 3' }}>
          <Chart option={main} figure={MAIN} />
        </div>
        {prior === undefined ? null : <Chart option={prior} figure={SQUARE} />}
        {likelihood === undefined ? null : <Chart option={likelihood} figure={SQUARE} />}
        {posterior === undefined ? null : <Chart option={posterior} figure={SQUARE} />}
      </div>
      <dl className="grid grid-cols-2 gap-x-8 gap-y-3 text-sm xl:grid-cols-1">
        {facts.map(([k, v]) => (
          <div key={k}>
            <dt className="text-xs text-muted">{k}</dt>
            <dd className="font-medium tabular-nums">{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}
