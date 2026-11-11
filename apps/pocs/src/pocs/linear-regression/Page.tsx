import { linearExample, linearHiddenKnobs } from 'omai-engine/bayesian-examples'
import type { ChainDiagnostics, PosteriorComparison, PosteriorPairs } from 'omai-engine/exposition'
import { band, bars, cartesian, lines, points } from 'omai-render/charts'
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
  selection: 'Which coefficients are used',
  pairs: 'Pair plot of the coefficients',
  curve: 'Chains or objective',
  numbers: 'The coefficients',
  check: 'How we know it works',
}
const FIG = figure({ width: 640, height: 380 })
const TOP = figure({ width: 640, height: 240 })
const BOTTOM = figure({ width: 640, height: 200 })
// The reference when the noise is learned: a long NUTS run of the same problem, in a worker of its own.
const REFERENCE = { method: 'nuts', numChains: 2, numWarmup: 300, numDraws: 500, thin: 1 }
const ORANGE = '#e8743b'

interface Fit {
  readonly seen: number
  readonly total: number
  readonly x: readonly number[]
  readonly mean: readonly number[]
  readonly sd: readonly number[]
  readonly lo: readonly number[]
  readonly hi: readonly number[]
  readonly predictiveLo: readonly number[]
  readonly predictiveHi: readonly number[]
  readonly lines: readonly (readonly number[])[]
  readonly truth: readonly number[]
  readonly data: { readonly x: readonly number[]; readonly y: readonly number[] }
  readonly coefficients: readonly {
    name: string
    truth: number | null
    mean: number
    sd: number
    exactMean: number | null
    exactSd: number | null
  }[]
}

interface Coefficients {
  readonly names: readonly string[]
  readonly truth: readonly number[]
  readonly mean: readonly number[]
  readonly sd: readonly number[]
  readonly lo: readonly number[]
  readonly hi: readonly number[]
  readonly score: readonly number[] | null
  readonly scoreLabel: string | null
  readonly active: readonly boolean[]
  readonly selection: { chosen: number; trulyActive: number; found: number; falseAlarms: number } | null
  readonly burnin: readonly number[] | null
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
      referenceFor={(config) => config['noiseModel'] !== 'known' || config['prior'] !== 'gaussian'}
      observe={(view) => <Outputs view={view} />}
      check={(view) => {
        const learned = view.config['noiseModel'] !== 'known' || view.config['prior'] !== 'gaussian'
        const prior = String(view.config['prior'])
        return (
          <ReferenceCheck
            view={view}
            probes={learned ? { reference: { path: 'linreg/draws' } } : {}}
            checkPath="linreg/check"
            explain={
              learned ? (
                <>
                  {prior === 'gaussian' ? 'With the noise learned' : 'With this prior'} there is no closed form, so the
                  reference is a long NUTS run (2 chains, 300 warm-up, 500 draws) of the same problem, computed in a
                  worker of its own once this run has ended. The check compares each coefficient (and the noise) of the
                  chosen method with it; the reference's own largest R-hat says whether it mixed (a spike and slab or an
                  ARD posterior can have several modes, and then the reference is only as good as that).
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
  const total = Math.max(2, Math.round(Number(config['rows'])))
  // The rows seen: all of them until the viewer scrubs back.
  const [asked, setAsked] = useState<number | null>(null)
  const seen = Math.max(2, Math.min(total, asked ?? total))
  const sparse = config['truth'] === 'sparse'
  const fit = useProbe(
    live,
    t,
    !sparse && (panels.active('fit') || panels.active('numbers')) ? 'linreg/fit' : undefined,
    {
      seen,
    } as never,
  ).value as Fit | null | undefined
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
  const learned = config['noiseModel'] !== 'known'
  const coefficients = useProbe(
    live,
    t,
    panels.active('selection') || (sparse && panels.active('numbers')) ? 'linreg/coefficients' : undefined,
    {
      burnin: true,
    } as never,
  ).value as Coefficients | null | undefined
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
        {sparse ? (
          <p className="text-sm text-muted">
            The sparse truth has many features rather than one input, so there is no curve to draw: see the coefficients
            below.
          </p>
        ) : fit === undefined || fit === null ? (
          <Waiting done={done} />
        ) : (
          <FitChart fit={fit} />
        )}
        <p className={`mt-1 text-xs text-muted ${sparse ? 'hidden' : ''}`}>
          The orange line is the posterior mean curve, the dark band its 95% credible band and the light band the 95%
          band of a new observation (the curve's uncertainty plus the noise); the thin lines are curves drawn from the
          posterior and the dashed line is the truth. The method is run again on just the rows seen: move the slider and
          the band contracts as data accumulate.
        </p>
      </FigurePanel>
      <FigurePanel panels={panels} name="selection">
        {coefficients === undefined || coefficients === null ? (
          <Waiting done={done} />
        ) : (
          <SelectionChart coefficients={coefficients} />
        )}
        <p className="mt-1 text-xs text-muted">
          The top chart is each coefficient's posterior mean with its 95% interval beside its true value (a dot after
          the name marks the coefficients that are truly not zero). For the spike and slab prior the lower chart is the
          posterior probability that the coefficient is in the slab; for ARD and the horseshoe it is the relevance, the
          typical share of the coefficient the data determine rather than the prior (1 for a coefficient the prior
          leaves alone, near 0 for one that is switched off). A coefficient scoring at least a half counts as selected.
          {coefficients?.selection == null
            ? ''
            : ` Selected ${coefficients.selection.chosen}: ${coefficients.selection.found} of the ${coefficients.selection.trulyActive} truly non-zero, ${coefficients.selection.falseAlarms} false.`}
          {coefficients?.burnin == null
            ? ''
            : ' The third bar is the same run with the ARD sparsity switched off, the smooth fit a burn-in would give first: the methods take no schedule, so raise the ARD sparsity knob from 0 by hand to watch the coefficients switch off.'}
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
                sparse
                  ? (coefficients?.names ?? []).map(
                      (name, j) =>
                        [
                          `${name}: true, posterior mean (sd)`,
                          `${fmt(coefficients?.truth[j], 3)}, ${fmt(coefficients?.mean[j], 3)} (${fmt(coefficients?.sd[j], 3)})`,
                        ] as [string, string],
                    )
                  : fit === undefined || fit === null
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

function SelectionChart(props: { readonly coefficients: Coefficients }): ReactNode {
  const { coefficients: c } = props
  const mode = useColourMode()
  const labels = c.names.map((n, j) => (c.active[j] ? `${n} \u25CF` : n))
  const top = useMemo(() => {
    const groups = [
      { name: 'true value', values: [...c.truth], colour: 'muted' },
      {
        name: 'posterior mean (95% interval)',
        values: [...c.mean],
        intervals: c.lo.map((l, j) => [l, c.hi[j] ?? l] as [number, number]),
      },
      ...(c.burnin === null ? [] : [{ name: 'burn-in fit (sparsity off)', values: [...c.burnin] }]),
    ]
    return cartesian([bars(labels, groups, { mode })], {
      mode,
      figure: TOP,
      x: { label: '' },
      y: { label: 'coefficient' },
    })
  }, [c, labels, mode])
  const bottom = useMemo(() => {
    if (c.score === null) return undefined
    return cartesian(
      [bars(labels, [{ name: c.scoreLabel ?? 'score', values: [...c.score] }], { mode, colourOffset: 3 })],
      {
        mode,
        figure: BOTTOM,
        x: { label: '' },
        y: { label: c.scoreLabel ?? 'score', limits: [0, 1] },
      },
    )
  }, [c, labels, mode])
  return (
    <div className="flex flex-col gap-2">
      <Chart option={top} figure={TOP} />
      {bottom === undefined ? (
        <p className="text-xs text-muted">The normal prior has no inclusion probability or relevance to show.</p>
      ) : (
        <Chart option={bottom} figure={BOTTOM} />
      )}
    </div>
  )
}
