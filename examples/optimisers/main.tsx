/**
 * README example "An optimiser leaves a trace": three optimisers on the Rosenbrock function, each run recording its
 * parameters and gradients at every step. The figure is drawn from those tapes, not from separate arrays.
 */
import { adam, minimizer, sgd } from 'mdml-engine/optim'
import { add, mul, square, sub, take, tensor, type Value } from 'mdml-engine/tensor'
import { policies, type Run, recorder, runAlgorithm } from 'mdml-engine/trace'
import { cartesian } from 'mdml-render/charts'
import { optimisationPaths, parameterPath } from 'mdml-render/trace'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'An optimiser leaves a trace',
  summary: 'Three optimisers on the Rosenbrock function; the paths over its contours are drawn from the recorded runs.',
  readme: true,
  alt: 'Three optimisers on the Rosenbrock function: their paths over the contours',
}

const rosenbrock = (p: Value) => add(square(sub(1, take(p, 0))), mul(10, square(sub(take(p, 1), square(take(p, 0))))))

export function run(): Outcome<Record<string, Run>> {
  const optimisers = {
    'gradient-descent': sgd({ learningRate: 0.01 }),
    momentum: sgd({ learningRate: 0.005, momentum: 0.9 }),
    adam: adam({ learningRate: 0.1 }),
  }
  const tapes: Record<string, Run> = {}
  for (const [name, optimiser] of Object.entries(optimisers)) {
    const tape = recorder()
    const start = tensor([-1.2, 1])
    runAlgorithm(
      minimizer((p) => rosenbrock(p as Value), start, optimiser),
      {
        seed: 0,
        maxSteps: 120,
        tracer: tape,
        policy: policies.teaching,
      },
    )
    tapes[name] = tape.run()
  }
  return { result: tapes, tapes }
}

export function View({ result }: { result: Record<string, Run> }) {
  const paths = Object.entries(result).map(([name, tape]) => ({
    name: name.replace('-', ' '),
    points: parameterPath(tape, 'params.x'),
  }))
  const layers = optimisationPaths((x, y) => (1 - x) ** 2 + 10 * (y - x * x) ** 2, paths, {
    x: [-1.6, 1.4],
    y: [-0.6, 1.8],
    n: 60,
    levels: 14,
  })
  return (
    <Figure width={640} height={400} title={meta.alt}>
      <Panel
        width={640}
        height={400}
        still={cartesian(layers, { x: { label: 'x' }, y: { label: 'y' }, legend: true })}
      />
    </Figure>
  )
}
