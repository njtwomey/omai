import type { ChainDiagnostics, PosteriorPairs } from 'omai-engine/exposition'
import { band, cartesian, histogram, lines } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import {
  ChainDiagnosticsView,
  Chart,
  FigurePanel,
  PosteriorPairsView,
  useColourMode,
  useProbe,
} from 'omai-render/react'
import { type ReactNode, useMemo } from 'react'
import { mixtureHiddenKnobs } from '../../../examples/index.ts'
import {
  ExamplePage,
  type ExampleView,
  fmt,
  Objective,
  ReferenceCheck,
  Table,
  useRecord,
  Waiting,
} from '../../../shared/ExamplePage.tsx'
import { mixtureExample, mixtureProducer } from './method.ts'

const PANELS = {
  fit: 'The data and the fitted mixture',
  pairs: 'Pair plot of the means, a weight and the spread',
  curve: 'Chains or objective',
  numbers: 'The components, sorted by mean',
  check: 'How we know it works',
}
const FIG = figure({ width: 640, height: 360 })
// The reference: a long NUTS run of the same data with the means ordered, in a worker of its own.
const REFERENCE = { method: 'nuts', numChains: 2, numWarmup: 300, numDraws: 500, thin: 1 }

interface Fit {
  readonly x: readonly number[]
  readonly data: readonly number[]
  readonly label: string
  readonly mean: readonly number[]
  readonly lo: readonly number[]
  readonly hi: readonly number[]
  readonly components: readonly (readonly number[])[]
  readonly truth: readonly number[] | null
  readonly rows: readonly {
    name: string
    mean: number
    sd: number
    lo: number
    hi: number
    truth: number | null
  }[]
  readonly outOfOrder: number
  readonly draws: number
}

export default function GaussianMixturePage(): ReactNode {
  return (
    <ExamplePage
      id="gaussian-mixture"
      example={mixtureExample as never}
      producer={mixtureProducer}
      panels={PANELS}
      exclude={mixtureHiddenKnobs}
      reference={REFERENCE}
      observe={(view) => <Outputs view={view} />}
      check={(view) => (
        <ReferenceCheck
          view={view}
          probes={{ reference: { path: 'mixture/summary' } }}
          checkPath="mixture/check"
          explain={
            <>
              The raw means of a run with free means cannot be compared with anything: the model does not tell the
              components apart, so which one a chain calls "mean 1" is arbitrary, a chain can swap them, and two chains
              can sit in different modes. What does not depend on the labels is compared instead: the components sorted
              by their means, and the predictive density. The reference is a long NUTS run (2 chains, 300 warm-up, 500
              draws) of the same data with the means ordered, so it has one mode, in a worker of its own once this run
              has ended. Each sorted mean, weight and spread of the chosen run is put against it in the reference's
              spreads, and the two predictive densities by the total variation distance between them (0 identical, 1
              disjoint).
            </>
          }
          rows={(c) => [
            ['sorted parameters compared', String(c['compared'])],
            [
              'largest difference of sorted means, in reference sds',
              c['maxMeanDifference'] === null ? '-' : fmt(c['maxMeanDifference']),
            ],
            [
              'largest difference of any parameter, in reference sds',
              `${fmt(c['maxMeanError'])} (${String(c['worst'])})`,
            ],
            ['median ratio of spreads (method / reference)', fmt(c['medianSdRatio'])],
            ['predictive densities: total variation distance', fmt(c['densityDistance'], 4)],
            ['draws in the reference', String(c['referenceDraws'])],
          ]}
        />
      )}
    />
  )
}

function Outputs(props: { readonly view: ExampleView }): ReactNode {
  const { live, t, panels, done, objective } = props.view
  const fit = useProbe(live, t, panels.active('fit') || panels.active('numbers') ? 'mixture/fit' : undefined).value as
    | Fit
    | null
    | undefined
  const pairs = useRecord<PosteriorPairs>(
    live,
    t,
    panels.active('pairs') ? 'mixture/pairs' : undefined,
    'posteriorPairs',
  )
  const diagnostics = useRecord<ChainDiagnostics>(
    live,
    t,
    panels.active('curve') ? 'mixture/diagnostics' : undefined,
    'chainDiagnostics',
  )
  return (
    <div className="flex flex-col gap-4">
      <FigurePanel panels={panels} name="fit">
        {fit === undefined || fit === null ? <Waiting done={done} /> : <FitChart fit={fit} />}
        <p className="mt-1 text-xs text-muted">
          The bars are the data. The solid line is the posterior mean of the predictive density with its 90% band, the
          thin lines the posterior mean of each component (sorted by mean, so they do not depend on the labels), and the
          dashed line the density the data were generated from, when it is known. Change the separation of the true
          components, or their number, and see where the mixture stops being able to tell them apart.
        </p>
      </FigurePanel>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <FigurePanel panels={panels} name="pairs">
          {pairs === undefined ? <Waiting done={done} /> : <PosteriorPairsView record={pairs} controls={false} />}
          <p className="mt-1 text-xs text-muted">
            The coordinates as the run labelled them. With the means ordered there is one mode. With free means there
            are K! symmetric ones: chains (colours) that sit in different modes, or one that swaps its labels, put mass
            at both mean 1 below mean 2 and the reverse. The crosses mark the truth when the means are ordered and K is
            the true number.
          </p>
        </FigurePanel>
        <div className="flex flex-col gap-4">
          <FigurePanel panels={panels} name="numbers">
            {fit === undefined || fit === null ? (
              <Waiting done={done} />
            ) : (
              <Table
                rows={[
                  ...fit.rows.map(
                    (r) =>
                      [
                        `${r.name}${r.truth === null ? '' : ' (truth)'}`,
                        `${fmt(r.mean, 2)} (${fmt(r.sd, 2)})${r.truth === null ? '' : ` ... ${fmt(r.truth, 2)}`}`,
                      ] as [string, string],
                  ),
                  ['draws with the means out of order', `${fmt(100 * fit.outOfOrder, 0)}% of ${fit.draws}`],
                ]}
              />
            )}
            <p className="mt-1 text-xs text-muted">
              Posterior mean (sd) of each component after sorting by mean, in the units of the data. Draws out of order
              are the label switching: 0% with ordered means, and with free means the share of draws in which the
              component called 1 has the larger mean.
            </p>
          </FigurePanel>
          <FigurePanel panels={panels} name="curve">
            {diagnostics !== undefined ? (
              <>
                <ChainDiagnosticsView record={diagnostics} />
                <p className="mt-1 text-xs text-muted">
                  In the model's standardised units. With free means, chains in different modes show up as a poor R-hat
                  of the means: the chains agree on the density, not on the labels.
                </p>
              </>
            ) : objective !== undefined ? (
              <Objective live={live} path={objective.path} label={objective.label} t={t} />
            ) : (
              <p className="text-sm text-muted">
                No chains or objective to show for this method: the samplers that keep chains (NUTS, HMC, tempering)
                show them once they have draws, the variational methods their objective.
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
    const x = [...fit.x]
    const lo = fit.x[0] ?? 0
    const hi = fit.x[fit.x.length - 1] ?? 1
    return cartesian(
      [
        histogram([...fit.data], {
          bins: Math.max(10, Math.min(40, Math.round(Math.sqrt(fit.data.length) * 1.5))),
          range: [lo, hi],
          density: true,
          colour: 'tab7',
          opacity: 0.45,
          mode,
        }),
        band([{ x, lower: [...fit.lo], upper: [...fit.hi] }], { mode }),
        lines(
          fit.components.map((c, j) => ({ x, y: [...c], name: `component ${j + 1}`, width: 1.5, opacity: 0.8 })),
          { mode, colourOffset: 2 },
        ),
        ...(fit.truth === null
          ? []
          : [
              lines([{ x, y: [...fit.truth], name: 'true density', dash: 'dashed', width: 2, colour: 'muted' }], {
                mode,
              }),
            ]),
        lines([{ x, y: [...fit.mean], name: 'predictive density', width: 3, colour: 'tab0' }], { mode }),
      ],
      { mode, figure: FIG, x: { label: fit.label }, y: { label: 'density' }, legend: true },
    )
  }, [fit, mode])
  return <Chart option={option} figure={FIG} />
}
