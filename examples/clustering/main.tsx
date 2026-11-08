/**
 * Example "A Gaussian mixture, step by step" (in the index, not in the README): expectation-maximisation for a mixture
 * of three Gaussians on overlapping blobs, recorded step by step. Everything drawn is read off the recording: the
 * log likelihood at each kept step, and the weights, means, covariances and responsibilities of the last kept step.
 * Left, the density the mixture gives the plane (a heat map and its contours, on the density scale) with each row
 * coloured by the component most responsible for it; right, the log likelihood climbing.
 */
import { generate } from 'mdml-engine/data'
import { blobs } from 'mdml-engine/datasets'
import { adjustedRandScore } from 'mdml-engine/metrics'
import { componentLabels, gaussianEmAlgorithm, initialMixture, mixturePosterior } from 'mdml-engine/mixture'
import { key } from 'mdml-engine/random'
import { toArray } from 'mdml-engine/tensor'
import { metricSeries, policies, type Run, recorder, runAlgorithm, tensorSeries } from 'mdml-engine/trace'
import { cartesian, lines } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'
import { axesOf, densityMap, gridOfValues, gridRows, windowOf } from '../_shared/maps.ts'

export const meta: ExampleModule['meta'] = {
  title: 'A Gaussian mixture, step by step',
  summary:
    'Expectation-maximisation for three overlapping Gaussians: the density, the log likelihood and the soft assignments.',
  readme: false,
  alt: 'A Gaussian mixture fitted by EM: the density it gives the plane with the rows coloured by component, and the log likelihood over the steps',
}

interface Result {
  readonly rows: number[][]
  /** The density of the mixture on a grid over the data's window. */
  readonly density: number[][]
  /** The most responsible component and the largest responsibility of each row at the last kept step. */
  readonly component: number[]
  readonly confidence: number[]
  readonly steps: number[]
  readonly logLikelihood: number[]
}

const GRID = 60

/** The values of a tensor channel at the last step it was kept, cut into rows of `width`. */
function lastRows(run: Run, name: string, width: number): number[][] {
  const flat = tensorSeries(run, name).at(-1)?.values ?? []
  return Array.from({ length: flat.length / width }, (_, i) => flat.slice(i * width, (i + 1) * width))
}

export function run(): Outcome<Result> {
  const { data } = generate(blobs, { knobs: { n: 100, centers: 3, clusterStd: 1.3, boxSize: 6 }, key: key(2) })
  const start = initialMixture(data.x, { nComponents: 3, covarianceType: 'full', init: 'random' }, key(0))
  const tape = recorder()
  // Every sixth step is kept (the tape stays small); the metrics and tensors share the schedule.
  const thin = {
    ...policies.teaching,
    roles: ['param', 'state'] as const,
    messages: false,
    frames: false,
    schedule: { every: 6 },
  }
  const fit = runAlgorithm(gaussianEmAlgorithm(data.x, start, { tol: 1e-5 }), {
    seed: 0,
    maxSteps: 100,
    tracer: tape,
    policy: thin,
  })
  const run = tape.run()

  // The mixture at the last kept step, as the recording holds it.
  const means = lastRows(run, 'means', 2)
  const k = means.length
  const mixture = {
    weights: lastRows(run, 'weights', k)[0] ?? [],
    means,
    covariances: lastRows(run, 'covariances', 4).map((r) => [
      [r[0] ?? 0, r[1] ?? 0],
      [r[2] ?? 0, r[3] ?? 0],
    ]),
  }
  const resp = lastRows(run, 'responsibilities', k)
  const rows = toArray(data.x) as number[][]
  const grid = gridRows(axesOf(windowOf(rows), GRID))
  const density = gridOfValues(
    mixturePosterior(grid, mixture).logNorm.map((v) => Math.exp(v)),
    GRID,
  )
  const logged = metricSeries(run, 'logLikelihood')
  const ari = adjustedRandScore(data.y, componentLabels(fit.state.resp)).estimate.value
  const confidence = resp.map((r) => Math.max(...r))
  const notes = [
    `${fit.steps} steps, ${fit.status}; mean log likelihood ${logged.values[0]?.toFixed(3)} at the first kept step, ${fit.state.logLikelihood.toFixed(3)} at the end`,
    `adjusted Rand index against the true blobs ${ari.toFixed(3)}; ${confidence.filter((c) => c < 0.9).length} of ${rows.length} rows have no component above 0.9`,
  ]
  return {
    result: {
      rows,
      density,
      component: componentLabels(resp),
      confidence,
      steps: logged.steps,
      logLikelihood: logged.values,
    },
    tapes: { run },
    notes,
  }
}

export function View({ result }: { result: Result }) {
  const window = windowOf(result.rows)
  const left = densityMap({
    window,
    field: result.density,
    label: 'density',
    rows: result.rows,
    groups: result.component,
    names: ['component 0', 'component 1', 'component 2'],
    width: 400,
    height: 340,
    xLabel: 'x0',
    yLabel: 'x1',
  })
  const small = figure({ width: 300, height: 340, padding: { top: 30, right: 14, bottom: 38, left: 56 } })
  const middle = cartesian(
    [lines([{ x: result.steps, y: result.logLikelihood, name: 'log likelihood', markers: true }])],
    { figure: small, x: { label: 'EM step' }, y: { label: 'mean log likelihood' }, legend: false },
  )
  return (
    <Figure width={720} height={366} title={meta.alt}>
      <Panel x={0} width={420} height={340} title="density and most responsible component" still={left} />
      <Panel x={420} width={300} height={340} title="log likelihood" still={middle} />
    </Figure>
  )
}
