/**
 * README example "Optimisers, first and second order" (in the README): the same start on the Rosenbrock function run
 * with three first-order methods (gradient descent, momentum, Adam) and three that use curvature (BFGS, L-BFGS,
 * Newton). Each run is recorded; the left panel draws the paths over the function (a heat map of log10(1 + f) filling
 * the axes, with its contours), and the right panel the loss against the step on a log axis, both read from the tapes.
 */
import { adam, bfgs, lbfgs, minimizer, newton, sgd } from 'omai-engine/optim'
import { add, mul, square, sub, take, tensor, toFlat, type Value } from 'omai-engine/tensor'
import { iterate, metricSeries, type Run, recorder, runAlgorithm, tracePolicy } from 'omai-engine/trace'
import { cartesian, lines, points } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { parameterPath } from 'omai-render/trace'
import type { Mode } from '../_shared/colours.ts'
import type { ExampleModule, Live, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'
import { axesOf, fieldLayers, gridOfValues, gridRows, windowOf } from '../_shared/maps.ts'

export const meta: ExampleModule['meta'] = {
  title: 'Optimisers, first and second order',
  summary:
    'Gradient descent, momentum and Adam against BFGS, L-BFGS and Newton on the Rosenbrock function: the paths and the loss from the recorded runs.',
  readme: true,
  alt: 'Six optimisers on the Rosenbrock function: their paths over the function and the loss against the step on a log axis',
  home: { label: 'Optimisers', fps: 10, seconds: 17 },
}

const rosenbrock = (p: Value): Value =>
  add(square(sub(1, take(p, 0))), mul(10, square(sub(take(p, 1), square(take(p, 0))))))
const f = (x: number, y: number): number => (1 - x) ** 2 + 10 * (y - x * x) ** 2

/** The methods in the order they are drawn, with their colours (first order dashed, curvature solid). */
const METHODS = [
  { key: 'gradient-descent', name: 'gradient descent', colour: 'tab0', curvature: false },
  { key: 'momentum', name: 'momentum', colour: 'tab4', curvature: false },
  { key: 'adam', name: 'Adam', colour: 'tab9', curvature: false },
  { key: 'bfgs', name: 'BFGS', colour: 'tab2', curvature: true },
  { key: 'lbfgs', name: 'L-BFGS', colour: 'tab3', curvature: true },
  { key: 'newton', name: 'Newton', colour: 'tab1', curvature: true },
] as const

/** Everything the figure draws, read from the tapes. */
interface Result {
  readonly methods: readonly {
    readonly name: string
    readonly colour: string
    readonly curvature: boolean
    readonly path: [number, number][]
    readonly steps: number[]
    readonly loss: number[]
  }[]
}

const START = [-1.2, 1] as const
const FIRST_ORDER_STEPS = 250
const TOLERANCE = 1e-6
/** Losses below this are drawn at it (the log axis cannot show zero). */
const FLOOR = 1e-12

const first = {
  'gradient-descent': () => sgd({ learningRate: 0.01 }),
  momentum: () => sgd({ learningRate: 0.005, momentum: 0.9 }),
  adam: () => adam({ learningRate: 0.1 }),
}
const second = { bfgs, lbfgs, newton }
const record = tracePolicy({ tensors: true, roles: ['param'], detail: 'full' })

/** The first step at which the recorded loss is at most the tolerance, or undefined. */
function stepsToTolerance(tape: Run): number | undefined {
  const { steps, values } = metricSeries(tape, 'loss')
  const i = values.findIndex((v) => v <= TOLERANCE)
  return i < 0 ? undefined : steps[i]
}

export function run(): Outcome<Result> {
  const tapes: Record<string, Run> = {}
  const notes: string[] = []
  const methods: Result['methods'][number][] = []
  for (const m of METHODS) {
    const tape = recorder()
    let path = 'x'
    if (!m.curvature) {
      path = 'params.x'
      const optimiser = first[m.key as keyof typeof first]()
      runAlgorithm(
        minimizer((p) => rosenbrock(p as Value), tensor([...START]), optimiser),
        { seed: 0, maxSteps: FIRST_ORDER_STEPS, tracer: tape, policy: record },
      )
    } else {
      second[m.key as keyof typeof second](rosenbrock, [...START], { tracer: tape, policy: record })
    }
    const run = tape.run()
    tapes[m.key] = run
    const { steps, values } = metricSeries(run, 'loss')
    methods.push({
      name: m.name,
      colour: m.colour,
      curvature: m.curvature,
      path: parameterPath(run, path),
      steps,
      loss: values,
    })
    // How long each takes to reach the tolerance: the first-order methods are run on until they do (untraced).
    const reached =
      stepsToTolerance(run) ??
      (m.curvature
        ? undefined
        : (() => {
            const long = recorder()
            runAlgorithm(
              minimizer((p) => rosenbrock(p as Value), tensor([...START]), first[m.key as keyof typeof first]()),
              { seed: 0, maxSteps: 5000, tracer: long, policy: tracePolicy({}) },
            )
            return stepsToTolerance(long.run())
          })())
    notes.push(`${m.name}: loss at most ${TOLERANCE} after ${reached ?? 'more than 5000'} steps`)
  }
  return { result: { methods }, tapes, notes }
}

/**
 * Stepped live, as it is watched: the three first-order methods take a step of their own `minimizer` algorithm each
 * time (from `iterate`), and the three that use curvature, which the library runs to the end in one call, are run
 * when it starts (they take about twenty steps) and shown one step at a time beside them.
 */
export function live(): Live<Result> {
  const firsts = METHODS.filter((m) => !m.curvature).map((m) => ({
    m,
    steps: iterate(
      minimizer((p) => rosenbrock(p as Value), tensor([...START]), first[m.key as keyof typeof first]()),
      { seed: 0, maxSteps: FIRST_ORDER_STEPS },
    ),
    path: [[START[0], START[1]]] as [number, number][],
    steps_: [0],
    loss: [f(START[0], START[1])],
  }))
  const seconds = METHODS.filter((m) => m.curvature).map((m) => {
    const tape = recorder()
    second[m.key as keyof typeof second](rosenbrock, [...START], { tracer: tape, policy: record })
    const run = tape.run()
    const series = metricSeries(run, 'loss')
    return { m, path: parameterPath(run, 'x'), steps_: series.steps, loss: series.values }
  })
  for (const x of firsts) x.steps.next() // the first state is the start, already in the path
  let t = 0
  const run: Live<Result> = {
    step: () => {
      let more = false
      // Two steps a frame (`frames` below counts frames of two steps).
      for (let k = 0; k < 2; k++) {
        t++
        for (const x of firsts) {
          const next = x.steps.next()
          if (next.done === true) continue
          const p = Array.from(toFlat(next.value.state.params as Value))
          x.path.push([p[0] ?? 0, p[1] ?? 0])
          x.steps_.push(t)
          x.loss.push(next.value.state.value)
          more = more || next.value.status === undefined
        }
      }
      return more || t < 30
    },
    result: () => ({
      methods: [
        ...firsts.map((x) => ({
          name: x.m.name,
          colour: x.m.colour,
          curvature: false,
          path: x.path,
          steps: x.steps_,
          loss: x.loss,
        })),
        ...seconds.map((x) => {
          const keep = x.steps_.filter((k) => k <= t).length
          return {
            name: x.m.name,
            colour: x.m.colour,
            curvature: true,
            path: x.path.slice(0, keep),
            steps: x.steps_.slice(0, keep),
            loss: x.loss.slice(0, keep),
          }
        }),
      ],
    }),
  }
  return run
}

/** Played in the browser: each frame is two steps of every method. */
export const frames = (result: Result): number =>
  Math.ceil(Math.max(...result.methods.map((m) => m.steps.at(-1) ?? 0)) / 2)

export function View({ result, at, mode = 'light' }: { result: Result; at?: number; mode?: Mode }) {
  // The limits come from the whole run, so the axes stay put while the paths grow.
  const all = result.methods.flatMap((m) => m.path)
  const step = at === undefined ? Number.POSITIVE_INFINITY : at * 2
  const upTo = (m: Result['methods'][number]): { path: [number, number][]; steps: number[]; loss: number[] } => {
    const keep = m.steps.filter((t) => t <= step).length
    return { path: m.path.slice(0, keep), steps: m.steps.slice(0, keep), loss: m.loss.slice(0, keep) }
  }
  const window = windowOf([...all, [1, 1]], 0.05)
  const n = 80
  const axes = axesOf(window, n)
  const field = gridOfValues(
    gridRows(axes).map(([x, y]) => Math.log10(1 + f(x ?? 0, y ?? 0))),
    n,
  )
  const one = (m: Result['methods'][number], dash: boolean) => ({
    x: upTo(m).path.map((p) => p[0]),
    y: upTo(m).path.map((p) => p[1]),
    name: m.name,
    colour: m.colour,
    width: 1.8,
    ...(dash ? { dash: 'dashed' as const } : {}),
  })
  const paths = cartesian(
    [
      ...fieldLayers({ window, field, label: 'log10(1 + f)', colourBar: false, levels: 10, mode }),
      lines(
        result.methods.map((m) => one(m, !m.curvature)),
        { mode },
      ),
      points(
        [
          {
            x: [START[0]],
            y: [START[1]],
            colour: 'fg',
            size: 9,
            symbol: 'emptyCircle',
            name: '__start',
            outline: false,
          },
        ],
        { mode },
      ),
      points([{ x: [1], y: [1], colour: 'fg', size: 9, symbol: 'diamond', name: '__minimum', outline: false }], {
        mode,
      }),
    ],
    {
      mode,
      figure: figure({ width: 520, height: 400, padding: { top: 12, right: 12, bottom: 38, left: 42 } }),
      x: { label: 'x', limits: [...window.x] },
      y: { label: 'y', limits: [...window.y] },
      legend: false,
    },
  )
  const loss = cartesian(
    [
      lines(
        result.methods.map((m) => ({
          x: upTo(m).steps,
          y: upTo(m).loss.map((v) => Math.max(v, FLOOR)),
          name: m.name,
          colour: m.colour,
          width: 1.8,
          ...(m.curvature ? {} : { dash: 'dashed' as const }),
        })),
        { mode },
      ),
    ],
    {
      mode,
      figure: figure({ width: 520, height: 400, padding: { top: 40, right: 12, bottom: 38, left: 56 } }),
      x: { label: 'step', limits: [0, FIRST_ORDER_STEPS] },
      y: { label: 'loss f (floored at 1e-12)', scale: 'log', limits: [FLOOR, 10] },
      legend: true,
    },
  )
  return (
    <Figure width={1040} height={426} title={meta.alt} mode={mode}>
      <Panel x={0} width={520} height={400} title="paths (dashed: first order, solid: uses curvature)" still={paths} />
      <Panel x={520} width={520} height={400} title="loss against step" still={loss} />
    </Figure>
  )
}
