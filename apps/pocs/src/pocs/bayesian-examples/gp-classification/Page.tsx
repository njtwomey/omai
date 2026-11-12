import type { ChainDiagnostics, PosteriorComparison, PosteriorPairs } from 'omai-engine/exposition'
import { band, cartesian, contours, heatmap, lines, points } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import {
  ChainDiagnosticsView,
  FigurePanel,
  PosteriorComparisonView,
  PosteriorPairsView,
  ResponsiveChart,
  useColourMode,
  useProbe,
} from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'
import { gpClassificationExample, gpHiddenKnobs } from '../../../examples/index.ts'
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
import { gpClassificationProducer } from './method.ts'

const PANELS = {
  fit: 'Predictive probability',
  latent: 'Latent function or uncertainty',
  pairs: 'Pair plot',
  curve: 'Chains or objective',
  numbers: 'The numbers',
  check: 'How we know it works',
}
// The reference: a long NUTS run of the same problem (the latent function whitened, so no funnel), in a worker of its own.
const REFERENCE = { method: 'nuts', numChains: 2, numWarmup: 150, numDraws: 250, thin: 1 }
const ORANGE = '#e8743b'

interface Bands {
  readonly mean: readonly number[]
  readonly lo: readonly number[]
  readonly hi: readonly number[]
  readonly sd: readonly number[]
}

interface Fit {
  readonly dim: 1 | 2
  readonly size: number
  readonly x: readonly number[]
  readonly y: readonly number[]
  readonly probability: Bands
  readonly latent: Bands
  readonly truth: readonly number[] | null
  readonly data: { readonly x: readonly (readonly number[])[]; readonly y: readonly number[] }
  readonly extent: readonly (readonly [number, number])[]
  readonly draws: number
  readonly flipped: number
}

interface Scores {
  readonly trainAccuracy: number
  readonly trainLogLikelihood: number
  readonly meanAbsError: number | null
  readonly ell: { mean: number; sd: number } | null
  readonly variance: { mean: number; sd: number } | null
  readonly truthLengthScale: number | null
  readonly draws: number
  readonly maxRhat: number | null
  readonly minEss: number | null
  readonly divergences: number | null
}

interface Extra {
  readonly name: string
  readonly vsMethod: number
  readonly vsReference: number
  readonly probabilityDifference: number
}

export default function GpClassificationPage(): ReactNode {
  return (
    <ExamplePage
      id="gp-classification"
      example={gpClassificationExample}
      producer={gpClassificationProducer}
      panels={PANELS}
      exclude={gpHiddenKnobs}
      reference={REFERENCE}
      observe={(view) => <Outputs view={view} />}
      check={(view) => (
        <ReferenceCheck
          view={view}
          probes={{ reference: { path: 'gpc/draws' } }}
          checkPath="gpc/check"
          explain={
            <>
              The posterior has no closed form, so the reference is a long NUTS run (2 chains, 150 warm-up, 250 draws)
              of the same problem, written with the latent function whitened (f = sqrt(variance) L z, z standard normal)
              so the sampler sees no funnel; it runs in a worker of its own once this run has ended. The check compares
              the length scale and the variance (when inferred) and the latent function at every input with it, in units
              of the reference spread. With the numbers fixed, the Laplace and EP fits of the library's gp domain are
              compared too: how far each is from this method and from the NUTS run.
            </>
          }
          unfinished="The reference run did not reach its draws within its own budget (20 s): lower the rows, fix the length scale and variance, or use an easier kernel."
          rows={(c) => [
            ['largest mean error (in reference sds)', fmt(c['maxMeanError'])],
            ['median spread ratio (method / reference)', fmt(c['medianSdRatio'])],
            ['largest Wasserstein distance (in sds)', fmt(c['maxWasserstein'])],
            ...((c['hyper'] as { name: string; meanError: number; sdRatio: number }[]) ?? []).map(
              (h): [string, string] => [
                `${h.name}: mean error (sds), spread ratio`,
                `${fmt(h.meanError, 2)}, ${fmt(h.sdRatio, 2)}`,
              ],
            ),
            ...((c['extras'] as Extra[]) ?? []).flatMap((e): [string, string][] => [
              [
                `${e.name}: latent mean, largest difference from this method / from NUTS (sds)`,
                `${fmt(e.vsMethod, 2)} / ${fmt(e.vsReference, 2)}`,
              ],
              [`${e.name}: largest probability difference from this method`, fmt(e.probabilityDifference, 3)],
            ]),
            ['reference draws', String(c['referenceDraws'])],
          ]}
          more={(c) => <PosteriorComparisonView record={c['comparison'] as PosteriorComparison} />}
        />
      )}
    />
  )
}

function Outputs(props: { readonly view: ExampleView }): ReactNode {
  const { live, t, panels, done, objective } = props.view
  const fit = useProbe(live, t, panels.active('fit') || panels.active('latent') ? 'gpc/fit' : undefined).value as
    | Fit
    | null
    | undefined
  const pairs = useRecord<PosteriorPairs>(live, t, panels.active('pairs') ? 'gpc/pairs' : undefined, 'posteriorPairs')
  const diagnostics = useRecord<ChainDiagnostics>(
    live,
    t,
    panels.active('curve') ? 'gpc/diagnostics' : undefined,
    'chainDiagnostics',
  )
  const scores = useProbe(live, t, panels.active('numbers') ? 'gpc/scores' : undefined).value as
    | Scores
    | null
    | undefined
  const ready = fit !== undefined && fit !== null
  const one = ready && fit.dim === 1
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <FigurePanel
          panels={panels}
          name="fit"
          title={one ? 'Predictive probability' : 'Predictive probability over the plane'}
        >
          {!ready ? (
            <Waiting done={done} />
          ) : one ? (
            <ProbabilityChart fit={fit} />
          ) : (
            <SurfaceChart fit={fit} spread={false} />
          )}
          <p className="mt-1 text-xs text-muted">
            {one
              ? 'The orange line is the posterior mean of the probability of a 1, the band its 95% credible band over the posterior draws, the dashed line the true probability (what the labels were drawn from, after the flips) and the dots the labels. Far from the data the band widens to the prior.'
              : 'The colour is the posterior mean probability of a 1 (blue to red around one half) with the 0.5 contour in black and the labels as dots; the dashed contour is the truth where the data set has one.'}
          </p>
        </FigurePanel>
        <FigurePanel panels={panels} name="latent" title={one ? 'Latent function' : 'Uncertainty of the probability'}>
          {!ready ? <Waiting done={done} /> : one ? <LatentChart fit={fit} /> : <SurfaceChart fit={fit} spread />}
          <p className="mt-1 text-xs text-muted">
            {one
              ? 'The latent function f has the Gaussian process as its prior; the posterior mean and 95% band of f, before the link turns it into a probability.'
              : 'The posterior standard deviation of the probability: large where the boundary is unsure and away from the data.'}
          </p>
        </FigurePanel>
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <FigurePanel panels={panels} name="pairs">
          {pairs === undefined ? <Waiting done={done} /> : <PosteriorPairsView record={pairs} controls={false} />}
          <p className="mt-1 text-xs text-muted">
            The length scale and the variance trade off (a long length scale with a large variance and a short one with
            a small variance explain the same labels); the latent values at three inputs show how the function is pinned
            down there.
          </p>
        </FigurePanel>
        <div className="flex flex-col gap-4">
          <FigurePanel panels={panels} name="numbers">
            <Table
              rows={
                scores === null || scores === undefined
                  ? []
                  : [
                      ['training accuracy', fmt(scores.trainAccuracy)],
                      ['mean log likelihood of the labels', fmt(scores.trainLogLikelihood)],
                      ...(scores.meanAbsError === null
                        ? []
                        : [
                            ['mean |probability - truth| over the grid', fmt(scores.meanAbsError, 3)] as [
                              string,
                              string,
                            ],
                          ]),
                      ...(scores.ell === null
                        ? []
                        : [
                            [
                              `length scale: mean (sd)${scores.truthLengthScale === null ? '' : `, true ${fmt(scores.truthLengthScale, 2)}`}`,
                              `${fmt(scores.ell.mean, 2)} (${fmt(scores.ell.sd, 2)})`,
                            ] as [string, string],
                          ]),
                      ...(scores.variance === null
                        ? []
                        : [
                            [
                              'variance: mean (sd)',
                              `${fmt(scores.variance.mean, 2)} (${fmt(scores.variance.sd, 2)})`,
                            ] as [string, string],
                          ]),
                      ['draws used', String(scores.draws)],
                      ...(scores.maxRhat === null ? [] : [['max R-hat', fmt(scores.maxRhat, 3)] as [string, string]]),
                      ...(scores.minEss === null ? [] : [['min ESS', fmt(scores.minEss, 0)] as [string, string]]),
                      ...(scores.divergences === null
                        ? []
                        : [['divergences', String(scores.divergences)] as [string, string]]),
                    ]
              }
            />
          </FigurePanel>
          <FigurePanel panels={panels} name="curve">
            {diagnostics !== undefined ? (
              <ChainDiagnosticsView record={diagnostics} />
            ) : objective !== undefined ? (
              <Objective live={live} path={objective.path} label={objective.label} t={t} />
            ) : (
              <p className="text-sm text-muted">
                No curve for this method: a sampler shows its chains once it has draws, and EP and Pathfinder end in a
                few steps.
              </p>
            )}
          </FigurePanel>
        </div>
      </div>
    </div>
  )
}

const flat = (v: readonly number[]): number[] => [...v]

function ProbabilityChart(props: { readonly fit: Fit }): ReactNode {
  const { fit } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) =>
      cartesian(
        [
          band([{ x: flat(fit.x), lower: flat(fit.probability.lo), upper: flat(fit.probability.hi) }], { mode }),
          ...(fit.truth === null
            ? []
            : [
                lines(
                  [{ x: flat(fit.x), y: flat(fit.truth), name: 'truth', dash: 'dashed', width: 2, colour: 'muted' }],
                  { mode },
                ),
              ]),
          lines([{ x: flat(fit.x), y: flat(fit.probability.mean), name: 'posterior mean', width: 3, colour: ORANGE }], {
            mode,
          }),
          points(
            [0, 1].map((cls) => ({
              x: fit.data.x.flatMap((r, i) => (fit.data.y[i] === cls ? [r[0] ?? 0] : [])),
              y: fit.data.y.flatMap((v) => (v === cls ? [cls] : [])),
              name: `label ${cls}`,
              colour: cls === 0 ? 'tab0' : 'tab3',
              size: 6,
              opacity: 0.7,
            })),
            { mode },
          ),
        ],
        {
          mode,
          figure,
          title: `${fit.data.y.length} labels, ${fit.draws} draws`,
          legend: false,
          x: { label: 'x', limits: [fit.extent[0]?.[0] ?? 0, fit.extent[0]?.[1] ?? 1] },
          y: { label: 'probability of a 1', limits: [-0.05, 1.05] },
        },
      ),
    [fit, mode],
  )
  return <ResponsiveChart aspect={1.5} narrowAspect={1.2} build={build} />
}

function LatentChart(props: { readonly fit: Fit }): ReactNode {
  const { fit } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) =>
      cartesian(
        [
          band([{ x: flat(fit.x), lower: flat(fit.latent.lo), upper: flat(fit.latent.hi) }], { mode }),
          lines([{ x: flat(fit.x), y: flat(fit.latent.mean), name: 'posterior mean of f', width: 3, colour: ORANGE }], {
            mode,
          }),
          lines(
            [
              {
                x: flat(fit.x),
                y: flat(fit.x).map(() => 0),
                name: 'f = 0 (probability one half)',
                dash: 'dotted',
                width: 1,
                colour: 'muted',
              },
            ],
            { mode },
          ),
        ],
        {
          mode,
          figure,
          legend: false,
          x: { label: 'x', limits: [fit.extent[0]?.[0] ?? 0, fit.extent[0]?.[1] ?? 1] },
          y: { label: 'latent function f' },
        },
      ),
    [fit, mode],
  )
  return <ResponsiveChart aspect={1.5} narrowAspect={1.2} build={build} />
}

function SurfaceChart(props: { readonly fit: Fit; readonly spread: boolean }): ReactNode {
  const { fit, spread } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const size = fit.size
      const xs = Array.from({ length: size }, (_, i) => fit.x[i] ?? 0)
      const ys = Array.from({ length: size }, (_, j) => fit.y[j * size] ?? 0)
      const values = spread ? fit.probability.sd : fit.probability.mean
      const z = Array.from({ length: size }, (_, j) =>
        Array.from({ length: size }, (_, i) => values[j * size + i] ?? Number.NaN),
      )
      const truth = fit.truth
      const zt =
        truth === null
          ? null
          : Array.from({ length: size }, (_, j) =>
              Array.from({ length: size }, (_, i) => truth[j * size + i] ?? Number.NaN),
            )
      const dots = [0, 1].map((cls) => ({
        x: fit.data.x.flatMap((r, i) => (fit.data.y[i] === cls ? [r[0] ?? 0] : [])),
        y: fit.data.x.flatMap((r, i) => (fit.data.y[i] === cls ? [r[1] ?? 0] : [])),
        name: `label ${cls}`,
        colour: cls === 0 ? 'tab0' : 'tab3',
        size: 6,
        opacity: 0.9,
      }))
      return cartesian(
        [
          spread ? heatmap(z, { x: xs, y: ys, mode }) : heatmap(z, { x: xs, y: ys, range: [0, 1], centre: 0.5, mode }),
          contours(
            Array.from({ length: size }, (_, j) =>
              Array.from({ length: size }, (_, i) => fit.probability.mean[j * size + i] ?? Number.NaN),
            ),
            {
              x: xs,
              y: ys,
              levels: [0.5],
              labels: false,
              colour: 'fg',
              width: 2,
              mode,
            },
          ),
          ...(zt === null
            ? []
            : [
                contours(zt, {
                  x: xs,
                  y: ys,
                  levels: [0.5],
                  labels: false,
                  colour: 'fg',
                  width: 1,
                  dash: 'dashed',
                  mode,
                }),
              ]),
          points(dots, { mode }),
        ],
        {
          mode,
          figure,
          legend: false,
          tooltip: false,
          x: { limits: [fit.extent[0]?.[0] ?? 0, fit.extent[0]?.[1] ?? 1] },
          y: { limits: [fit.extent[1]?.[0] ?? 0, fit.extent[1]?.[1] ?? 1] },
        },
      )
    },
    [fit, spread, mode],
  )
  return <ResponsiveChart aspect={1.05} narrowAspect={1} build={build} />
}
