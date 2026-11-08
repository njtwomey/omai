/**
 * Example "A small neural network, trained and watched": a multilayer perceptron on the two moons, trained as a
 * session. The loss of every minibatch step is on the tape, and the session can be asked for the network as it was at
 * any step, so the figure shows the loss curve beside the decision surface at the start, in the middle and at the end.
 * The dataset carries its truth, so the best possible accuracy is known.
 */
import { generate } from 'mdml-engine/data'
import { moons } from 'mdml-engine/datasets'
import type { DecisionSurface, Exposition } from 'mdml-engine/exposition'
import { defineTrainer } from 'mdml-engine/learn'
import { crossEntropy } from 'mdml-engine/losses'
import { mlp } from 'mdml-engine/nn'
import { adam } from 'mdml-engine/optim'
import { key } from 'mdml-engine/random'
import { metricSeries, type Run } from 'mdml-engine/trace'
import { metricChart } from 'mdml-render/trace'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Row } from '../_shared/figure.tsx'
import { probabilitySurface } from '../_shared/probability-surface.ts'

export const meta: ExampleModule['meta'] = {
  title: 'A small neural network, trained and watched',
  summary:
    'A multilayer perceptron on the two moons: the loss at every step, and the decision surface at the start, the middle and the end.',
  readme: false,
  alt: 'The training loss of a small network on two moons, and its decision surface at the start, the middle and the end of training',
}

/** A classifier: a 2-6-6-2 network, cross-entropy, Adam, minibatches of 30 rows for 16 epochs (one session step is one epoch). */
const network = defineTrainer({
  name: 'moons-mlp',
  kind: 'classifier',
  defaults: {},
  model: (_data, _hyper, k) => mlp([2, 6, 6, 2], { key: k }),
  objective: () => (prediction, target) => crossEntropy(prediction, target),
  optimiser: (rate) => adam({ learningRate: rate }),
  settings: () => ({ learningRate: 0.05, epochs: 16, batchSize: 30, unit: 'epoch' }),
})

interface Result {
  readonly run: Run
  readonly surfaces: readonly { readonly step: number; readonly surface: Exposition }[]
}

export function run(): Outcome<Result> {
  const { data, truth } = generate(moons, { knobs: { n: 80, noise: 0.25 }, key: key(0) })
  const training = network.session(data, { key: key(1) })
  training.advance()
  const steps = [0, Math.floor(training.t / 2), training.t]
  const surfaces = steps.map((step) => ({
    step,
    surface: training.probe(step, 'model/decision-surface', { resolution: 36 }) as Exposition,
  }))
  const loss = metricSeries(training.run(), 'train/loss').values
  const accuracy = steps.map((t) => (training.fittedAt(t).score?.(data) ?? Number.NaN).toFixed(3))
  const notes = [
    `${training.t} steps; loss ${loss[0]?.toFixed(3)} at the start, ${loss.at(-1)?.toFixed(3)} at the end`,
    `accuracy at steps ${steps.join(', ')}: ${accuracy.join(', ')}; best possible ${(1 - (truth.bayesError ?? 0)).toFixed(3)}`,
  ]
  return { result: { run: training.run(), surfaces }, tapes: { session: training }, notes }
}

export function View({ result }: { result: Result }) {
  return (
    <Row
      alt={meta.alt}
      width={330}
      height={300}
      panels={[
        { title: 'training loss', still: metricChart(result.run, ['train/loss']) },
        ...result.surfaces.map(({ step, surface }) => ({
          title: `step ${step}`,
          still: probabilitySurface(surface as unknown as DecisionSurface, { legend: false }),
        })),
      ]}
    />
  )
}
