import { gam, s } from 'mdml-engine/gam'
import { choice, defaults, dict, int, real } from 'mdml-engine/space'
import type { Config } from 'mdml-engine/trace'
import { metricSeries } from 'mdml-engine/trace'
import { band, cartesian, lines, points } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import {
  Chart,
  ControlPanel,
  DatasetPicker,
  type DatasetSelection,
  type LiveSession,
  SessionPlayer,
  SpaceControls,
  selectEntry,
  useColourMode,
  useDataset,
  useProbe,
  useRun,
  useScrubClick,
  useSession,
  useSessionPreferences,
} from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { type CurveData, curveData, gridOf } from '../../shared/curve-data.ts'

const DATA = 'regression1d'
const FIGURE = figure({ width: 960, height: 420 })
const FIRST = (): DatasetSelection => selectEntry(DATA, 'sineCurve')
const STEPS = 40
// Points at which the fit is drawn: enough that steps and kinks are drawn as such.
const GRID = 301

const SPACE = dict({
  method: choice(['reml', 'gcv', 'fixed'], {
    default: 'reml',
    label: 'how the smoothing is chosen',
    descriptions: {
      reml: 'restricted maximum likelihood: the smoothing parameter that maximises the Laplace-approximate marginal likelihood (mgcv’s default)',
      gcv: 'generalised cross-validation: the smoothing parameter that minimises an estimate of the prediction error',
      fixed: 'no selection: use the smoothing parameter set below',
    },
  }),
  iteration: choice(['performance', 'outer'], {
    default: 'performance',
    label: 'iteration',
    descriptions: {
      performance:
        'Gu’s performance iteration: choose the smoothing parameter afresh at every reweighted least-squares step (exact for a Gaussian response)',
      outer:
        'Wood’s outer iteration: fit to convergence at each trial smoothing parameter and take Newton steps on the criterion, one per step here',
    },
  }),
  logLambda: real(-6, 6, { default: 0, label: 'log10 smoothing parameter (used when fixed)' }),
  k: int(4, 60, { default: 12, label: 'basis size' }),
  basis: choice(['ps', 'bs', 'cr', 'tp'], {
    default: 'ps',
    label: 'basis',
    descriptions: {
      ps: 'P-spline (Eilers and Marx): a B-spline basis of the chosen degree with a difference penalty of the chosen order on neighbouring coefficients',
      bs: 'B-spline: a B-spline basis of the chosen degree with a penalty on the spline (mgcv’s bs)',
      cr: 'cubic regression spline: a cubic spline fixed by its values at the knots, penalised on its integrated squared second derivative (no degree to choose)',
      tp: 'thin-plate regression spline: a low-rank form of the thin-plate spline, with no knots to place (no degree to choose)',
    },
  }),
  shape: choice(['none', 'increasing', 'decreasing', 'convex', 'concave'], {
    default: 'none',
    label: 'shape constraint (P-splines)',
    descriptions: {
      none: 'no constraint',
      increasing: 'the curve may only rise (a quadratic program at every step)',
      decreasing: 'the curve may only fall',
      convex: 'the curve may only bend upwards (second derivative at least zero)',
      concave: 'the curve may only bend downwards (second derivative at most zero)',
    },
  }),
  degree: int(0, 6, { default: 3, label: 'spline degree (0 constant, 1 linear, 2 quadratic, 3 cubic)' }),
  order: int(1, 3, { default: 2, label: 'penalty order' }),
})
type Controls = {
  method: string
  iteration: string
  logLambda: number
  k: number
  basis: string
  shape: string
  degree: number
  order: number
}
// What a basis has no use for: the controls left out.
const EXCLUDED: Readonly<Record<string, string[]>> = {
  ps: [],
  bs: ['order', 'shape'],
  cr: ['degree', 'order', 'shape'],
  tp: ['degree', 'order', 'shape'],
}
const START: Controls = {
  method: 'reml',
  iteration: 'performance',
  logLambda: 0,
  k: 12,
  basis: 'ps',
  shape: 'none',
  degree: 3,
  order: 2,
}

export default function GamPage(): ReactNode {
  const [selection, setSelection] = useState<DatasetSelection>(FIRST)
  const [controls, setControls] = useState<Controls>(START)
  const generated = useDataset(DATA, selection)
  const curve = useMemo(() => curveData(generated.data, generated.truth as never), [generated.data, generated.truth])
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 items-start gap-3 md:grid-cols-2">
        <ControlPanel title="data" columns={1}>
          <DatasetPicker collection={DATA} value={selection} onChange={setSelection} />
        </ControlPanel>
        <ControlPanel title="the smooth and its penalty" columns={1}>
          <SpaceControls
            space={SPACE}
            value={controls}
            exclude={EXCLUDED[controls.basis] ?? []}
            onChange={(v) => setControls(v as unknown as Controls)}
          />
          <p className="text-xs text-muted">
            The degree belongs to the B-spline bases (ps, bs); the penalty order and the shape constraint to ps. The
            cubic regression spline (cr) and thin-plate spline (tp) have a fixed form, so those controls are hidden for
            them.
          </p>
        </ControlPanel>
      </div>
      {curve === undefined ? (
        <p className="text-sm text-muted">This page needs a one-dimensional regression data set.</p>
      ) : (
        <Fit curve={curve} controls={controls} reset={() => setControls(START)} />
      )}
    </div>
  )
}

function Fit(props: { readonly curve: CurveData; readonly controls: Controls; readonly reset: () => void }): ReactNode {
  const { curve, controls, reset } = props
  const prefs = useSessionPreferences()
  const [at, setAt] = useState<number | null>(null)
  const method = useMemo(() => {
    const term = s(0, {
      k: controls.k,
      basis: controls.basis as never,
      ...(controls.basis === 'ps'
        ? {
            order: controls.order,
            degree: controls.degree,
            ...(controls.shape === 'none' ? {} : { shape: controls.shape as never }),
          }
        : {}),
      ...(controls.basis === 'bs' ? { degree: controls.degree } : {}),
    })
    const model = gam.with({
      terms: [term],
      method: controls.method as never,
      iteration: controls.iteration as never,
      ...(controls.method === 'fixed' ? { sp: [10 ** controls.logLambda] } : {}),
    })
    return model.method({ x: curve.x.map((v) => [v]), y: [...curve.y] })
  }, [curve, controls])
  const live = useSession(
    method,
    { config: defaults(method.config) as Config, mode: prefs.mode, steps: STEPS },
    [method],
    {
      hold: prefs.hold,
    },
  )
  return (
    <div className="flex flex-col gap-3">
      <View live={live} at={at} onScrub={setAt} curve={curve} />
      <SessionPlayer live={live} at={at} onScrub={setAt} onReset={reset} />
      <p className="max-w-3xl text-xs text-muted">
        Each step is one iteration of the fit: the smoothing parameter is re-chosen (or kept, when fixed) and the curve
        refitted. The shaded band is the 95% credible band. A larger smoothing parameter pulls the curve towards a
        straight line (fewer effective degrees of freedom); a smaller one lets it follow the noise. A Gaussian model
        settles in a few iterations; the “outer” iteration takes Newton steps on the smoothing parameter, one per step.
      </p>
    </div>
  )
}

function View(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the session is generic over the method
  readonly live: LiveSession<any, any>
  readonly at: number | null
  readonly onScrub: (t: number | null) => void
  readonly curve: CurveData
}): ReactNode {
  const { live, curve } = props
  const mode = useColourMode()
  const run = useRun(live)
  const t = Math.max(0, Math.min(props.at ?? live.head, live.head))
  const grid = useMemo(() => gridOf(curve.lo, curve.hi, GRID), [curve])
  const probed = useProbe(live, t, 'model/band', { x: grid.map((v) => [v]) })
  const params = useProbe(live, t, 'model/params')
  const steps = useMemo(() => Array.from({ length: live.head + 1 }, (_, i) => i), [live.head])
  const seek = useScrubClick(steps, live.head, props.onScrub)
  const deviance = useMemo(() => metricSeries(run, 'solver/deviance'), [run])
  const truthSorted = useMemo(() => curve.truthAt?.(grid), [curve, grid])
  const shown = probed.value as { fit: number[]; lower: number[]; upper: number[] } | undefined
  const option = useMemo(() => {
    const x = grid
    const pick = (v: number[]): number[] => v
    return cartesian(
      [
        ...(shown === undefined ? [] : [band([{ x, lower: pick(shown.lower), upper: pick(shown.upper) }], { mode })]),
        points([{ x: [...curve.x], y: [...curve.y], name: 'data', opacity: 0.55 }], { mode, colourOffset: 7 }),
        ...(truthSorted === undefined
          ? []
          : [lines([{ x, y: truthSorted, name: 'truth', dash: 'dashed', width: 2, colour: 'muted' }], { mode })]),
        ...(shown === undefined ? [] : [lines([{ x, y: pick(shown.fit), name: 'GAM fit', width: 3 }], { mode })]),
      ],
      {
        mode,
        figure: FIGURE,
        x: { label: 'x', limits: [curve.lo, curve.hi] },
        y: { label: 'y', limits: [Math.min(...curve.y), Math.max(...curve.y)] },
      },
    )
  }, [shown, grid, curve, truthSorted, mode])
  const p = params.value as { logLambda?: number[]; edf?: number; scale?: number; deviance?: number } | undefined
  const facts: [string, string][] = [
    ['step', `${t} of ${live.head}`],
    ['smoothing parameter', (p?.logLambda ?? []).map((l) => Number(Math.exp(l).toPrecision(3)).toString()).join(', ')],
    ['effective degrees of freedom', p?.edf === undefined ? '' : p.edf.toFixed(2)],
    [
      'deviance',
      deviance.values[Math.min(t, deviance.values.length - 1)] === undefined
        ? ''
        : Number((deviance.values[Math.min(t, deviance.values.length - 1)] as number).toPrecision(4)).toString(),
    ],
  ]
  return (
    <div className="flex flex-col gap-3">
      <Chart option={option} figure={FIGURE} onInit={seek} />
      <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-4">
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
