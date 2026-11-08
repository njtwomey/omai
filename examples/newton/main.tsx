/**
 * Example "Second order: grad of grad, Hessians and Newton": `grad(grad(f))`, `hessian` and `hvp` from
 * `mdml-engine/autodiff`, then Newton's method against gradient descent on the Rosenbrock function. Both runs are
 * traced, and the figure draws their paths from the tapes, with the local quadratic model (the Hessian's ellipse) at
 * two points (scaled for display: every ellipse has the same long axis, and only its shape is the Hessian's).
 */
import { grad, hessian, hvp } from 'mdml-engine/autodiff'
import { minimize, newton, sgd } from 'mdml-engine/optim'
import { add, mul, square, sub, take, tensor, toArray, toFlat, type Value } from 'mdml-engine/tensor'
import { policies, type Run, recorder } from 'mdml-engine/trace'
import { cartesian, lines } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { parameterPath } from 'mdml-render/trace'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'
import { axesOf, fieldLayers, gridOfValues, gridRows, windowOf } from '../_shared/maps.ts'

export const meta: ExampleModule['meta'] = {
  title: 'Second order: grad of grad, Hessians and Newton',
  summary: 'grad(grad(f)), hessian and hvp; Newton reaches the minimum in 12 steps where gradient descent needs 2662.',
  readme: false,
  alt: "Newton's path against gradient descent on the Rosenbrock function, with the curvature at two points",
}

const rosenbrock = (p: Value): Value =>
  add(square(sub(1, take(p, 0))), mul(10, square(sub(take(p, 1), square(take(p, 0))))))
const cube = (x: Value): Value => mul(mul(x, x), x)

interface Result {
  readonly tapes: Record<string, Run>
  /** The Hessian at the points whose quadratic model is drawn. */
  readonly curvature: readonly { readonly at: [number, number]; readonly hessian: number[][]; readonly name: string }[]
}

export function run(): Outcome<Result> {
  const start = tensor([-1.2, 1])
  const second = recorder()
  const newtonRun = newton(rosenbrock, [-1.2, 1], { tracer: second, policy: policies.teaching })
  const first = recorder()
  minimize(rosenbrock, start, {
    method: sgd({ learningRate: 0.01 }),
    maxSteps: 60,
    tracer: first,
    policy: policies.teaching,
  })
  const descent = minimize(rosenbrock, start, { method: sgd({ learningRate: 0.01 }), maxSteps: 5000, gradTol: 1e-5 })

  const at = (p: [number, number]) => toArray(hessian(rosenbrock)(tensor(p))) as number[][]
  const tapes = { newton: second.run(), 'gradient-descent': first.run() }
  const notes = [
    `grad(grad(cube))(tensor(2)) = ${toFlat(grad(grad(cube))(tensor(2)))}`,
    `hessian(rosenbrock)(tensor([1, 1])) = ${JSON.stringify(at([1, 1]))}`,
    `hvp(rosenbrock, tensor([1, 1]), tensor([1, 0])) = ${toFlat(hvp(rosenbrock, tensor([1, 1]), tensor([1, 0])))}`,
    `newton: ${newtonRun.steps} steps (${newtonRun.status}); gradient descent: ${descent.steps} steps (${descent.status})`,
  ]
  const curvature = [
    { at: [-1.2, 1] as [number, number], hessian: at([-1.2, 1]), name: 'Hessian at start' },
    { at: [1, 1] as [number, number], hessian: at([1, 1]), name: 'Hessian at minimum' },
  ]
  return { result: { tapes, curvature }, tapes, notes }
}

/**
 * The ellipse of the local quadratic model, d' H d / 2 = level, drawn for display: its long semi-axis is set to `reach`
 * in the units of the plot whatever the curvature, so the ellipse stays on the plot. The short semi-axis keeps its
 * ratio to the long one (the square root of the ratio of the Hessian's eigenvalues), so the shape is the model's.
 */
function ellipse(p: [number, number], h: number[][], reach: number) {
  const [a = 0, b = 0] = h[0] ?? []
  const d = h[1]?.[1] ?? 0
  const mid = (a + d) / 2
  const spread = Math.sqrt(((a - d) / 2) ** 2 + b * b)
  const angle = 0.5 * Math.atan2(2 * b, a - d)
  const [small, large] = [mid + spread, mid - spread] // eigenvalues: the largest curvature gives the shortest axis
  const [u, v] = [reach * Math.sqrt(large / small), reach]
  const t = Array.from({ length: 61 }, (_, i) => (i / 60) * 2 * Math.PI)
  return {
    x: t.map((k) => p[0] + u * Math.cos(k) * Math.cos(angle) - v * Math.sin(k) * Math.sin(angle)),
    y: t.map((k) => p[1] + u * Math.cos(k) * Math.sin(angle) + v * Math.sin(k) * Math.cos(angle)),
  }
}

const REACH = 0.35

export function View({ result }: { result: Result }) {
  const steps = parameterPath(result.tapes['newton'] as Run, 'x')
  const descent = parameterPath(result.tapes['gradient-descent'] as Run, 'params.x')
  const shapes = result.curvature.map((c) => ellipse(c.at, c.hessian, REACH))
  const all = [...steps, ...descent, ...shapes.flatMap((e) => e.x.map((x, i): [number, number] => [x, e.y[i] ?? 0]))]
  const window = windowOf(all, 0.05)
  const n = 80
  const axes = axesOf(window, n)
  const field = gridOfValues(
    gridRows(axes).map(([x, y]) => Math.log10(1 + (1 - (x ?? 0)) ** 2 + 10 * ((y ?? 0) - (x ?? 0) ** 2) ** 2)),
    n,
  )
  const colours = ['tab3', 'tab2']
  const layers = [
    ...fieldLayers({ window, field, label: 'log10(1 + f)', colourBar: false, levels: 10 }),
    lines([
      {
        x: descent.map((p) => p[0]),
        y: descent.map((p) => p[1]),
        name: 'gradient descent',
        colour: 'tab0',
        width: 1.8,
        dash: 'dashed',
      },
      { x: steps.map((p) => p[0]), y: steps.map((p) => p[1]), name: 'Newton', colour: 'tab1', width: 2, markers: true },
      ...result.curvature.map((c, i) => ({
        ...(shapes[i] as { x: number[]; y: number[] }),
        name: c.name,
        colour: colours[i] ?? 'fg',
        width: 2,
      })),
    ]),
  ]
  const plot = cartesian(layers, {
    figure: figure({ width: 680, height: 420, padding: { top: 30, right: 12, bottom: 38, left: 42 } }),
    x: { label: 'x', limits: [...window.x] },
    y: { label: 'y', limits: [...window.y] },
    legend: true,
  })
  return (
    <Figure width={680} height={420} title={meta.alt}>
      <Panel width={680} height={420} still={plot} />
    </Figure>
  )
}
