/**
 * Example "A small neural network, trained and watched": a multilayer perceptron on the two moons, trained as a
 * session. The loss of every minibatch step is on the tape, and the session can be asked for the network as it was at
 * any step, so the figure shows the loss curve beside the decision surface at the start, in the middle and at the end.
 * The dataset carries its truth, so the best possible accuracy is known.
 */
import { generate } from 'omai-engine/data'
import { moons } from 'omai-engine/datasets'
import type { DecisionSurface, Exposition } from 'omai-engine/exposition'
import { defineTrainer } from 'omai-engine/learn'
import { crossEntropy } from 'omai-engine/losses'
import { mlp } from 'omai-engine/nn'
import { adam } from 'omai-engine/optim'
import { key } from 'omai-engine/random'
import { metricSeries, type Run } from 'omai-engine/trace'
import { metricChart } from 'omai-render/trace'
import type { Mode } from '../_shared/colours.ts'
import type { ExampleModule, Live, Outcome } from '../_shared/contract.ts'
import { Row } from '../_shared/figure.tsx'
import { probabilitySurface } from '../_shared/probability-surface.ts'

export const meta: ExampleModule['meta'] = {
  title: 'A small neural network, trained and watched',
  summary:
    'A multilayer perceptron on the two moons: the loss at every step, and the decision surface at the start, the middle and the end.',
  readme: false,
  alt: 'The training loss of a small network on two moons, and its decision surface at the start, the middle and the end of training',
  home: { label: 'Neural network', seconds: 18 },
}

/** A classifier: a 2-16-16-2 network, cross-entropy, Adam, minibatches of 20 rows for 150 epochs (one session step is one epoch). */
const network = defineTrainer({
  name: 'moons-mlp',
  kind: 'classifier',
  defaults: {},
  model: (_data, _hyper, k) => mlp([2, 16, 16, 2], { key: k }),
  objective: () => (prediction, target) => crossEntropy(prediction, target),
  optimiser: (rate) => adam({ learningRate: rate }),
  settings: () => ({ learningRate: 0.02, epochs: 150, batchSize: 20, unit: 'epoch' }),
})

interface Result {
  readonly run: Run
  readonly surfaces: readonly { readonly step: number; readonly accuracy: number; readonly surface: Exposition }[]
}

/** How many snapshots of the network are read back from the session: the frames of the animation. */
const SNAPSHOTS = 60

export function run(): Outcome<Result> {
  const { data, truth } = generate(moons, { knobs: { n: 200, noise: 0.25 }, key: key(0) })
  // A step budget longer than the training keeps the session's checkpoints few, so the tape stays small.
  const training = network.session(data, { key: key(1), steps: 20000 })
  training.advance()
  // Snapshots closer together early on, where the network changes fastest.
  const steps = Array.from({ length: SNAPSHOTS }, (_, i) => Math.round(training.t * (i / (SNAPSHOTS - 1)) ** 1.8))
  const surfaces = steps.map((step) => ({
    step,
    accuracy: training.fittedAt(step).score?.(data) ?? Number.NaN,
    surface: training.probe(step, 'model/decision-surface', { resolution: 48 }) as Exposition,
  }))
  const loss = metricSeries(training.run(), 'train/loss').values
  const accuracy = [0, 0.07, 0.3, 1].map((share) => {
    const t = Math.round(training.t * share)
    return (training.fittedAt(t).score?.(data) ?? Number.NaN).toFixed(3)
  })
  const notes = [
    `${training.t} steps; loss ${loss[0]?.toFixed(3)} at the start, ${loss.at(-1)?.toFixed(3)} at the end`,
    `accuracy at 0, 7, 30 and 100% of the steps: ${accuracy.join(', ')}; best possible ${(1 - (truth.bayesError ?? 0)).toFixed(3)}`,
  ]
  return { result: { run: training.run(), surfaces }, tapes: { session: training }, notes }
}

/**
 * Trained live, as it is watched: each step is one epoch of the same session `run` uses, and the surface of the network
 * after it is read back (`model/decision-surface`) as the snapshot to draw.
 */
export function live(): Live<Result> {
  const { data } = generate(moons, { knobs: { n: 200, noise: 0.25 }, key: key(0) })
  const training = network.session(data, { key: key(1), steps: 20000 })
  const surfaces: { step: number; accuracy: number; surface: Exposition }[] = []
  const snapshot = (): void => {
    surfaces.push({
      step: training.t,
      accuracy: Number.NaN,
      surface: training.probe(training.t, 'model/decision-surface', { resolution: 44 }) as Exposition,
    })
  }
  snapshot()
  return {
    step: () => {
      training.advance({ steps: 1 })
      snapshot()
      return training.status === undefined
    },
    result: () => ({ run: training.run(), surfaces }),
  }
}

/** Played in the browser: one frame per snapshot of the network. */
export const frames = (result: Result): number => result.surfaces.length - 1

/** The snapshot nearest to a share of the training. */
const nearest = (result: Result, share: number): Result['surfaces'][number] => {
  const last = result.surfaces.at(-1)?.step ?? 0
  return result.surfaces.reduce((best, s) =>
    Math.abs(s.step - share * last) < Math.abs(best.step - share * last) ? s : best,
  )
}

export function View({ result, at, mode = 'light' }: { result: Result; at?: number; mode?: Mode }) {
  if (at !== undefined) {
    const now = result.surfaces[Math.min(at, result.surfaces.length - 1)]
    if (now === undefined) throw new Error('neural-network: no snapshot to draw')
    const run: Run = {
      ...result.run,
      events: result.run.events.filter((e) => !('step' in e) || (e.step ?? 0) <= now.step),
    }
    return (
      <Row
        mode={mode}
        alt={meta.alt}
        width={380}
        height={340}
        panels={[
          { title: 'training loss', still: metricChart(run, ['train/loss'], { mode }) },
          {
            title: Number.isFinite(now.accuracy)
              ? `step ${now.step}: accuracy ${now.accuracy.toFixed(3)}`
              : `step ${now.step}`,
            still: probabilitySurface(now.surface as unknown as DecisionSurface, { legend: false, mode }),
          },
        ]}
      />
    )
  }
  const shown = [0, 0.07, 0.3, 1].map((share) => nearest(result, share))
  return (
    <Row
      mode={mode}
      alt={meta.alt}
      width={300}
      height={300}
      panels={[
        { title: 'training loss', still: metricChart(result.run, ['train/loss'], { mode }) },
        ...shown.map(({ step, accuracy, surface }) => ({
          title: `step ${step}: accuracy ${accuracy.toFixed(3)}`,
          still: probabilitySurface(surface as unknown as DecisionSurface, { legend: false, mode }),
        })),
      ]}
    />
  )
}
