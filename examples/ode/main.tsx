/**
 * Example "Predators and prey, from an equation": the Lotka-Volterra equations (prey grow and are eaten, predators eat and
 * die) solved as an initial value problem by `solveIvp`, a traced algorithm that takes one adaptive step at a time. The
 * figure has the two populations over time, the same solution as a path in the (prey, predators) plane, and the step size
 * the solver chose at each step, read back from the tape.
 */
import { lotkaVolterraField } from 'omai-engine/dynamics'
import { solveIvp } from 'omai-engine/dynamics/ode'
import { tensor, toFlat } from 'omai-engine/tensor'
import { metricSeries, type Run, recorder, tracePolicy } from 'omai-engine/trace'
import { cartesian, lines, points } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'Predators and prey, from an equation',
  summary:
    'The Lotka-Volterra differential equations solved with an adaptive ODE solver: the two populations, the loop they trace, and the step sizes the solver chose.',
  readme: false,
  alt: 'Prey and predator populations over time, the closed loop they trace in the prey and predator plane, and the solver step size against time',
}

const RATES = { alpha: 1, beta: 0.1, delta: 0.075, gamma: 1.5 }
const END = 20

interface Result {
  readonly t: number[]
  readonly prey: number[]
  readonly predators: number[]
  /** The time of each accepted step and the step size it took. */
  readonly stepTime: number[]
  readonly stepSize: number[]
  /** The balance point: prey gamma/delta, predators alpha/beta. */
  readonly equilibrium: readonly [number, number]
}

/** The quantity the equations keep constant, `delta x - gamma ln x + beta y - alpha ln y`. */
const conserved = (x: number, y: number): number =>
  RATES.delta * x - RATES.gamma * Math.log(x) + RATES.beta * y - RATES.alpha * Math.log(y)

export function run(): Outcome<Result> {
  const tape = recorder()
  const solution = solveIvp(lotkaVolterraField(RATES), [0, END], tensor([40, 9]), {
    method: 'RK45',
    tEval: Array.from({ length: 401 }, (_, i) => (i * END) / 400), // dense output, so the curves are smooth
    rtol: 1e-6,
    atol: 1e-8,
    tracer: tape,
    policy: tracePolicy({ metrics: ['t', 'h'], schedule: { every: 1 } }),
  })
  const states = solution.ys.map((s) => Array.from(toFlat(s)))
  const prey = states.map((s) => s[0] ?? 0)
  const predators = states.map((s) => s[1] ?? 0)
  const recorded: Run = tape.run()
  const times = metricSeries(recorded, 't').values
  const sizes = metricSeries(recorded, 'h').values
  const taken = sizes.flatMap((h, i) => (h > 0 ? [i] : []))
  const drift = Math.max(
    ...prey.map((x, i) => Math.abs(conserved(x, predators[i] ?? 1) - conserved(prey[0] ?? 1, predators[0] ?? 1))),
  )
  return {
    result: {
      t: [...solution.t],
      prey,
      predators,
      stepTime: taken.map((i) => times[i] ?? 0),
      stepSize: taken.map((i) => sizes[i] ?? 0),
      equilibrium: [RATES.gamma / RATES.delta, RATES.alpha / RATES.beta],
    },
    tapes: { solve: recorded },
    notes: [
      `${solution.status}: ${solution.nSteps} accepted steps, ${solution.nRejected} rejected`,
      `the conserved quantity drifts by at most ${drift.toExponential(2)} over t = 0 to ${END}`,
      `step sizes from ${Math.min(...taken.map((i) => sizes[i] ?? 0)).toFixed(4)} to ${Math.max(...taken.map((i) => sizes[i] ?? 0)).toFixed(4)}`,
    ],
  }
}

const PAD = { top: 12, right: 14, bottom: 40, left: 52 }

export function View({ result }: { result: Result }) {
  const populations = cartesian(
    [
      lines([
        { x: result.t, y: result.prey, name: 'prey', colour: 'tab0', width: 2 },
        { x: result.t, y: result.predators, name: 'predators', colour: 'tab1', width: 2 },
      ]),
    ],
    {
      figure: figure({ width: 400, height: 340, padding: PAD }),
      x: { label: 'time' },
      y: { label: 'population' },
      legend: true,
    },
  )
  const loop = cartesian(
    [
      lines([{ x: result.prey, y: result.predators, name: 'solution', colour: 'tab0', width: 2 }]),
      points([
        {
          x: [result.equilibrium[0]],
          y: [result.equilibrium[1]],
          name: 'balance point',
          colour: 'tab3',
          size: 11,
          opacity: 1,
        },
      ]),
    ],
    {
      figure: figure({ width: 360, height: 340, padding: PAD }),
      x: { label: 'prey' },
      y: { label: 'predators' },
      legend: true,
    },
  )
  const steps = cartesian(
    [lines([{ x: result.stepTime, y: result.stepSize, name: 'step size', colour: 'tab2', width: 2, markers: true }])],
    {
      figure: figure({ width: 340, height: 340, padding: PAD }),
      x: { label: 'time' },
      y: { label: 'step size' },
      legend: false,
    },
  )
  return (
    <Figure width={1100} height={366} title={meta.alt}>
      <Panel x={0} width={400} height={340} title="the two populations over time" still={populations} />
      <Panel x={400} width={360} height={340} title="the same solution as a path" still={loop} />
      <Panel x={760} width={340} height={340} title="the step size the solver chose" still={steps} />
    </Figure>
  )
}
