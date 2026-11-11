import { linearExample, linearHiddenKnobs } from 'omai-engine/bayesian-examples'
import type { ChainDiagnostics, PosteriorComparison, PosteriorPairs } from 'omai-engine/exposition'
import { band, cartesian, lines, points } from 'omai-render/charts'
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
import { linearRegressionProducer } from './method.ts'

const PANELS = {
  fit: 'The fit after the rows seen',
  pairs: 'Pair plot of the coefficients',
  curve: 'Chains or objective',
  numbers: 'The coefficients',
  check: 'How we know it works',
}
const FIG = figure({ width: 640, height: 380 })
// The reference when the noise is learned: a long NUTS run of the same problem, in a worker of its own.
const REFERENCE = { method: 'nuts', numChains: 2, numWarmup: 300, numDraws: 500, thin: 1 }
const ORANGE = '#e8743b'

interface Fit {
  readonly seen: number
  readonly total: number
  readonly x: readonly number[]
  readonly mean: readonly number[]
  readonly lo: readonly number[]
  readonly hi: readonly number[]
  readonly predictiveLo: readonly number[]
  readonly predictiveHi: readonly number[]
  readonly lines: readonly (readonly number[])[]
  readonly truth: readonly number[]
  readonly data: { readonly x: readonly number[]; readonly y: readonly number[] }
  readonly coefficients: readonly {
    name: string
    truth: number
    mean: number
    sd: number
    exactMean: number | null
    exactSd: number | null
  }[]
}

export default function LinearRegressionPage(): ReactNode {
  return (
    <ExamplePage
      id="linear-regression"
      example={linearExample}
      producer={linearRegressionProducer}
      panels={PANELS}
      exclude={linearHiddenKnobs}
      reference={REFERENCE}
      referenceFor={(config) => config['noiseModel'] === 'learned'}
      observe={(view) => <Outputs view={view} />}
      check={(view) => {
        const learned = view.config['noiseModel'] === 'learned'
        return (
          <ReferenceCheck
            view={view}
            probes={learned ? { reference: { path: 'linreg/draws' } } : {}}
            checkPath="linreg/check"
            explain={
              learned ? (
                <>
                  With the noise learned there is no closed form, so the reference is a long NUTS run (2 chains, 300
                  warm-up, 500 draws) of the same problem, computed in a worker of its own once this run has ended. The
                  check compares each coefficient (and the noise) of the chosen method with it.
                </>
              ) : (
                <>
                  With the noise known, a normal prior on the coefficients and a normal likelihood give a normal
                  posterior in closed form: precision (1/prior scale squared) I + X'X / noise squared. The chosen
                  method's draws are compared with that exact normal, coefficient by coefficient.
                </>
              )
            }
            rows={(c) => [
              ['largest mean error (in reference sds)', fmt(c['maxMeanError'])],
              ['median spread ratio (method / reference)', fmt(c['medianSdRatio'])],
              ['largest Wasserstein distance (in sds)', fmt(c['maxWasserstein'])],
              ['largest KL (reference || method)', fmt(c['maxKl'], 4)],
              ['checked against', c['against'] === 'exact' ? 'the exact normal posterior' : 'a long NUTS run'],
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
  const total = Math.max(2, Math.round(Number(config['rows'])))
  // The rows seen: all of them until the viewer scrubs back.
  const [asked, setAsked] = useState<number | null>(null)
  const seen = Math.max(2, Math.min(total, asked ?? total))
  const fit = useProbe(live, t, panels.active('fit') || panels.active('numbers') ? 'linreg/fit' : undefined, {
    seen,
  } as never).value as Fit | null | undefined
  const pairs = useRecord<PosteriorPairs>(
    live,
    t,
    panels.active('pairs') ? 'linreg/pairs' : undefined,
    'posteriorPairs',
  )
  const diagnostics = useRecord<ChainDiagnostics>(
    live,
    t,
    panels.active('curve') ? 'linreg/diagnostics' : undefined,
    'chainDiagnostics',
  )
  const learned = config['noiseModel'] === 'learned'
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
          The orange line is the posterior mean curve, the dark band its 95% credible band and the light band the 95%
          band of a new observation (the curve's uncertainty plus the noise); the thin lines are curves drawn from the
          posterior and the dashed line is the truth. The method is run again on just the rows seen: move the slider and
          the band contracts as data accumulate.
        </p>
      </FigurePanel>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <FigurePanel panels={panels} name="pairs">
          {pairs === undefined ? <Waiting done={done} /> : <PosteriorPairsView record={pairs} controls={false} />}
          <p className="mt-1 text-xs text-muted">
            The correlation of the intercept and the slope is what a line with inputs away from zero costs: choose the
            offset inputs and the two trade off; centred inputs decouple them.
          </p>
        </FigurePanel>
        <div className="flex flex-col gap-4">
          <FigurePanel panels={panels} name="numbers">
            <Table
              rows={
                fit === undefined || fit === null
                  ? []
                  : fit.coefficients.flatMap((c) => [
                      [
                        `${c.name}: true, posterior mean (sd)`,
                        `${fmt(c.truth, 3)}, ${fmt(c.mean, 3)} (${fmt(c.sd, 3)})`,
                      ] as [string, string],
                      ...(c.exactMean === null
                        ? []
                        : [
                            [`${c.name}: exact mean (sd)`, `${fmt(c.exactMean, 3)} (${fmt(c.exactSd, 3)})`] as [
                              string,
                              string,
                            ],
                          ]),
                    ])
              }
            />
            {learned ? null : (
              <p className="mt-1 text-xs text-muted">The noise is known: its true value is used in the model.</p>
            )}
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
    return cartesian(
      [
        band([{ x: [...fit.x], lower: [...fit.predictiveLo], upper: [...fit.predictiveHi] }], { mode }),
        band([{ x: [...fit.x], lower: [...fit.lo], upper: [...fit.hi] }], { mode, colourOffset: 1 }),
        lines(
          fit.lines.map((l) => ({ x: [...fit.x], y: [...l], colour: ORANGE, width: 1, opacity: 0.3 })),
          { mode },
        ),
        points([{ x: [...fit.data.x], y: [...fit.data.y], name: 'data', opacity: 0.85 }], { mode, colourOffset: 7 }),
        lines([{ x: [...fit.x], y: [...fit.truth], name: 'truth', dash: 'dashed', width: 2, colour: 'muted' }], {
          mode,
        }),
        lines([{ x: [...fit.x], y: [...fit.mean], name: 'posterior mean', width: 3, colour: ORANGE }], { mode }),
      ],
      {
        mode,
        figure: FIG,
        title: `after ${fit.seen} of ${fit.total} rows`,
        x: { label: 'x' },
        y: { label: 'y', limits: [lo - pad, hi + pad] },
      },
    )
  }, [fit, mode])
  return <Chart option={option} figure={FIG} />
}
