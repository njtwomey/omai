import type { ChainDiagnostics, PosteriorComparison, PosteriorPairs } from 'omai-engine/exposition'
import { band, cartesian, histogram, lines, points } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import {
  ChainDiagnosticsView,
  Chart,
  FigurePanel,
  NumberField,
  PosteriorComparisonView,
  PosteriorPairsView,
  useColourMode,
  useProbe,
} from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { robustExample, robustHiddenKnobs } from '../../examples/index.ts'
import {
  ExamplePage,
  type ExampleView,
  fmt,
  Objective,
  ReferenceCheck,
  Table,
  useRecord,
  Waiting,
} from '../../shared/ExamplePage.tsx'
import { robustRegressionProducer } from './method.ts'

const PANELS = {
  fit: 'The fit after the rows seen',
  numbers: 'How far each fit is dragged',
  df: 'Degrees of freedom',
  pairs: 'Pair plot',
  curve: 'Chains or objective',
  check: 'How we know it works',
}
const FIG = figure({ width: 640, height: 380 })
const SMALL = figure({ width: 520, height: 260 })
// The reference for every likelihood but a Gaussian one with the noise known: a long NUTS run in a worker of its own.
const REFERENCE = { method: 'nuts', numChains: 2, numWarmup: 300, numDraws: 500, thin: 1 }
const ORANGE = '#e8743b'
const RED = '#d6334a'

interface Fit {
  readonly seen: number
  readonly total: number
  readonly likelihood: string
  readonly x: readonly number[]
  readonly mean: readonly number[]
  readonly lo: readonly number[]
  readonly hi: readonly number[]
  readonly predictiveLo: readonly number[]
  readonly predictiveHi: readonly number[]
  readonly lines: readonly (readonly number[])[]
  readonly truth: readonly number[]
  readonly gaussian: readonly number[]
  readonly inliers: readonly number[]
  readonly data: {
    readonly x: readonly number[]
    readonly y: readonly number[]
    readonly outlier: readonly boolean[]
  }
  readonly errors: { readonly method: number; readonly gaussian: number; readonly inliers: number }
  readonly coefficients: readonly {
    name: string
    truth: number | null
    mean: number
    sd: number
    gaussian: number | null
    inliers: number | null
  }[]
  readonly outliers: number
}

interface DegreesOfFreedom {
  readonly draws: readonly number[]
  readonly grid: readonly number[]
  readonly prior: readonly number[]
}

export default function RobustRegressionPage(): ReactNode {
  return (
    <ExamplePage
      id="robust-regression"
      example={robustExample}
      producer={robustRegressionProducer}
      panels={PANELS}
      exclude={robustHiddenKnobs}
      reference={REFERENCE}
      referenceFor={(config) => !(config['likelihood'] === 'gaussian' && config['noiseModel'] === 'known')}
      observe={(view) => <Outputs view={view} />}
      check={(view) => {
        const exact = view.config['likelihood'] === 'gaussian' && view.config['noiseModel'] === 'known'
        return (
          <ReferenceCheck
            view={view}
            probes={exact ? {} : { reference: { path: 'robust/draws' } }}
            checkPath="robust/check"
            explain={
              exact ? (
                <>
                  With a Gaussian likelihood and the noise known, a normal prior on the coefficients gives a normal
                  posterior in closed form: precision (1/prior scale squared) I + X'X / noise squared. The chosen
                  method's draws are compared with that exact normal, coefficient by coefficient (it is also the
                  Gaussian fit drawn grey in the figure).
                </>
              ) : (
                <>
                  With this likelihood there is no closed form, so the reference is a long NUTS run (2 chains, 300
                  warm-up, 500 draws) of the same problem, computed in a worker of its own once this run has ended. The
                  check compares each coefficient, the noise scale and the degrees of freedom of the chosen method with
                  it; the reference's own largest R-hat says whether it mixed. The truth is not the reference: a
                  heavy-tailed fit is checked against the same model's posterior, and its distance from the true line is
                  in the table above.
                </>
              )
            }
            rows={(c) => [
              ['largest mean error (in reference sds)', fmt(c['maxMeanError'])],
              ['median spread ratio (method / reference)', fmt(c['medianSdRatio'])],
              ['largest Wasserstein distance (in sds)', fmt(c['maxWasserstein'])],
              ['largest KL (reference || method)', fmt(c['maxKl'], 4)],
              ['checked against', c['against'] === 'exact' ? 'the exact normal posterior' : 'a long NUTS run'],
              ...(typeof c['referenceRhat'] === 'number'
                ? [['reference: largest R-hat', fmt(c['referenceRhat'], 3)] as [string, string]]
                : []),
            ]}
            more={(c) => <PosteriorComparisonView record={c['comparison'] as PosteriorComparison} />}
          />
        )
      }}
    />
  )
}

function Outputs(props: { readonly view: ExampleView }): ReactNode {
  const { live, t, panels, done, config, objective } = props.view
  const total = Math.max(5, Math.round(Number(config['rows'])))
  const [asked, setAsked] = useState<number | null>(null)
  const seen = Math.max(2, Math.min(total, asked ?? total))
  const fit = useProbe(live, t, panels.active('fit') || panels.active('numbers') ? 'robust/fit' : undefined, {
    seen,
  } as never).value as Fit | null | undefined
  const df = useProbe(live, t, panels.active('df') ? 'robust/df' : undefined).value as
    | DegreesOfFreedom
    | null
    | undefined
  const pairs = useRecord<PosteriorPairs>(
    live,
    t,
    panels.active('pairs') ? 'robust/pairs' : undefined,
    'posteriorPairs',
  )
  const diagnostics = useRecord<ChainDiagnostics>(
    live,
    t,
    panels.active('curve') ? 'robust/diagnostics' : undefined,
    'chainDiagnostics',
  )
  return (
    <div className="flex flex-col gap-4">
      <FigurePanel panels={panels} name="fit">
        <div className="mb-2 max-w-sm">
          <NumberField
            label="rows seen"
            value={seen}
            onChange={(v) => setAsked(Math.max(2, Math.min(total, Math.round(v))))}
            min={2}
            max={total}
            step={1}
            slider
          />
        </div>
        {fit === undefined || fit === null ? <Waiting done={done} /> : <FitChart fit={fit} />}
        <p className="mt-1 text-xs text-muted">
          Red crosses are the outliers, dots the inliers. The orange line is the posterior mean of the chosen likelihood
          with its 95% band (dark) and the 95% band of a new observation under that likelihood (light); the dashed line
          is the truth and the grey dotted line the Gaussian fit to the same rows (the exact normal posterior). A
          Gaussian likelihood sits on the grey line, pulled towards the outliers; a heavy-tailed one stays near the
          truth. Move the slider to fit fewer rows, and set the outliers to leverage points to see the slope dragged
          hardest.
        </p>
      </FigurePanel>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <FigurePanel panels={panels} name="numbers">
          {fit === undefined || fit === null ? (
            <Waiting done={done} />
          ) : (
            <div className="flex flex-col gap-3">
              <Table
                title={`${fit.outliers} outliers in ${fit.seen} rows: the curve's distance from the truth (rms)`}
                rows={[
                  [`this likelihood (${fit.likelihood})`, fmt(fit.errors.method)],
                  ['Gaussian fit to all rows', fmt(fit.errors.gaussian)],
                  ['Gaussian fit to the inliers only', fmt(fit.errors.inliers)],
                ]}
              />
              <Table
                title="coefficients: true, posterior mean (sd), Gaussian fit, inliers only"
                rows={fit.coefficients.map((c) => [
                  c.name,
                  `${c.truth === null ? '-' : fmt(c.truth, 2)}, ${fmt(c.mean, 2)} (${fmt(c.sd, 2)}), ${
                    c.gaussian === null ? '-' : fmt(c.gaussian, 2)
                  }, ${c.inliers === null ? '-' : fmt(c.inliers, 2)}`,
                ])}
              />
            </div>
          )}
        </FigurePanel>
        <FigurePanel panels={panels} name="df">
          {df === undefined ? (
            <Waiting done={done} />
          ) : df === null ? (
            <p className="text-sm text-muted">
              No degrees of freedom to infer: choose the Student-t likelihood with its degrees of freedom learned.
            </p>
          ) : (
            <DfChart df={df} />
          )}
          <p className="mt-1 text-xs text-muted">
            Few degrees of freedom (1 is a Cauchy) mean heavy tails: the data say how heavy. With no outliers the
            posterior piles up at large values (the t becomes a Gaussian); with outliers it moves down towards 1 to 3.
          </p>
        </FigurePanel>
        <FigurePanel panels={panels} name="pairs">
          {pairs === undefined ? <Waiting done={done} /> : <PosteriorPairsView record={pairs} controls={false} />}
        </FigurePanel>
        <FigurePanel panels={panels} name="curve">
          {diagnostics !== undefined ? (
            <ChainDiagnosticsView record={diagnostics} />
          ) : objective !== undefined ? (
            <Objective live={live} path={objective.path} label={objective.label} t={t} />
          ) : (
            <p className="text-sm text-muted">
              No curve for this method: a sampler shows its chains once it has draws.
            </p>
          )}
        </FigurePanel>
      </div>
    </div>
  )
}

function FitChart(props: { readonly fit: Fit }): ReactNode {
  const { fit } = props
  const mode = useColourMode()
  const option = useMemo(() => {
    const all = [...fit.data.y, ...fit.predictiveLo, ...fit.predictiveHi]
    const lo = Math.min(...all)
    const hi = Math.max(...all)
    const pad = 0.05 * (hi - lo || 1)
    const pick = (flag: boolean): { x: number[]; y: number[] } => ({
      x: fit.data.x.filter((_, i) => fit.data.outlier[i] === flag),
      y: fit.data.y.filter((_, i) => fit.data.outlier[i] === flag),
    })
    const inliers = pick(false)
    const outliers = pick(true)
    return cartesian(
      [
        band([{ x: [...fit.x], lower: [...fit.predictiveLo], upper: [...fit.predictiveHi] }], { mode }),
        band([{ x: [...fit.x], lower: [...fit.lo], upper: [...fit.hi] }], { mode, colourOffset: 1 }),
        lines(
          fit.lines.map((l) => ({ x: [...fit.x], y: [...l], colour: ORANGE, width: 1, opacity: 0.3 })),
          { mode },
        ),
        points([{ ...inliers, name: 'inliers', opacity: 0.85 }], { mode, colourOffset: 7 }),
        points([{ ...outliers, name: 'outliers', colour: RED, symbol: 'cross', size: 10 }], { mode }),
        lines([{ x: [...fit.x], y: [...fit.truth], name: 'truth', dash: 'dashed', width: 2, colour: 'muted' }], {
          mode,
        }),
        lines(
          [{ x: [...fit.x], y: [...fit.gaussian], name: 'Gaussian fit', dash: 'dotted', width: 2, colour: 'muted' }],
          {
            mode,
          },
        ),
        lines([{ x: [...fit.x], y: [...fit.mean], name: 'posterior mean', width: 3, colour: ORANGE }], { mode }),
      ],
      {
        mode,
        figure: FIG,
        title: `${fit.likelihood}: after ${fit.seen} of ${fit.total} rows`,
        x: { label: 'x' },
        y: { label: 'y', limits: [lo - pad, hi + pad] },
      },
    )
  }, [fit, mode])
  return <Chart option={option} figure={FIG} />
}

function DfChart(props: { readonly df: DegreesOfFreedom }): ReactNode {
  const { df } = props
  const mode = useColourMode()
  const option = useMemo(
    () =>
      cartesian(
        [
          histogram([...df.draws], { bins: 24, density: true, mode, name: 'posterior' }),
          lines([{ x: [...df.grid], y: [...df.prior], name: 'prior', dash: 'dashed', width: 2, colour: 'muted' }], {
            mode,
          }),
        ],
        { mode, figure: SMALL, x: { label: 'degrees of freedom' }, y: { label: 'density' } },
      ),
    [df, mode],
  )
  return <Chart option={option} figure={SMALL} />
}
