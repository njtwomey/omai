/**
 * Example "A Gaussian mixture, step by step" (in the index, not in the README): expectation-maximisation for a mixture
 * of three Gaussians on overlapping blobs, recorded step by step. A `mixtureSteps` record is read off the tape
 * (log likelihood, weights, means, covariances and responsibilities at every step) and drawn by the library's view:
 * the components as ellipses, the log likelihood climbing, the weights. The second panel colours each row by its
 * largest responsibility, so the rows between two components stand out. The blobs carry their truth, so the fit can be
 * scored against the labels that made them.
 */
import { generate } from 'mdml-engine/data'
import { blobs } from 'mdml-engine/datasets'
import { type Exposition, type MixtureSteps, mixtureSteps } from 'mdml-engine/exposition'
import { adjustedRandScore } from 'mdml-engine/metrics'
import { componentLabels, gaussianEmAlgorithm, initialMixture } from 'mdml-engine/mixture'
import { key } from 'mdml-engine/random'
import { toArray } from 'mdml-engine/tensor'
import { metricSeries, policies, type Run, recorder, runAlgorithm, tensorSeries } from 'mdml-engine/trace'
import { cartesian, points } from 'mdml-render/charts'
import { stillOf } from 'mdml-render/views'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'A Gaussian mixture, step by step',
  summary:
    'Expectation-maximisation for three overlapping Gaussians: the components, the log likelihood and the soft assignments.',
  readme: false,
  alt: 'A Gaussian mixture fitted by EM: components as ellipses with the log likelihood and weights, and each row coloured by its largest responsibility',
}

interface Result {
  readonly steps: MixtureSteps
  /** The largest responsibility of each row at the last step. */
  readonly confidence: readonly number[]
}

/** The values of a tensor channel at each step the log likelihood was logged at (not the start), cut into rows of `width`. */
function rowsOver(run: Run, name: string, width: number): number[][][] {
  const logged = new Set(metricSeries(run, 'logLikelihood').steps)
  return tensorSeries(run, name)
    .filter((e) => logged.has(e.step))
    .map((e) => {
      const flat = e.values ?? []
      return Array.from({ length: flat.length / width }, (_, i) => flat.slice(i * width, (i + 1) * width))
    })
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

  // Everything below is read off the recorded run, not off the final state.
  const means = rowsOver(run, 'means', 2)
  const k = means[0]?.length ?? 0
  const features = ['x0', 'x1'] as const
  const x = toArray(data.x) as number[][]
  const steps = mixtureSteps({
    family: 'gaussian',
    objective: metricSeries(run, 'logLikelihood').values,
    weights: rowsOver(run, 'weights', k).map((r) => r[0] ?? []),
    means,
    covariances: rowsOver(run, 'covariances', 4),
    responsibilities: rowsOver(run, 'responsibilities', k),
    rows: { points: x, features },
  })
  const last = steps.responsibilities?.at(-1) ?? []
  const confidence = last.map((r) => Math.max(...r))
  const ari = adjustedRandScore(data.y, componentLabels(fit.state.resp)).estimate.value
  const ll = steps.objective
  const uncertain = confidence.filter((c) => c < 0.9).length
  const notes = [
    `${fit.steps} steps, ${fit.status}; mean log likelihood ${ll[0]?.toFixed(3)} at the first kept step, ${fit.state.logLikelihood.toFixed(3)} at the end`,
    `adjusted Rand index against the true blobs ${ari.toFixed(3)}; ${uncertain} of ${confidence.length} rows have no component above 0.9`,
  ]
  return { result: { steps, confidence }, tapes: { run }, notes }
}

export function View({ result }: { result: Result }) {
  const pts = result.steps.rows?.points ?? []
  const soft = cartesian(
    [
      points([
        {
          x: pts.map((p) => p[0] ?? 0),
          y: pts.map((p) => p[1] ?? 0),
          values: [...result.confidence],
          size: 7,
          name: 'rows',
        },
      ]),
    ],
    { x: { label: 'x0' }, y: { label: 'x1' }, aspect: 'equal' },
  )
  return (
    <Figure width={1080} height={400} title={meta.alt}>
      <Panel
        x={0}
        width={640}
        height={374}
        title="components, log likelihood, weights"
        still={stillOf(result.steps as Exposition, { mode: 'light' })}
      />
      <Panel x={660} width={420} height={374} title="largest responsibility of each row" still={soft} />
    </Figure>
  )
}
