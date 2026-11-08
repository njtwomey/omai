/**
 * README example "Second order: grad of grad, Hessians and Newton": `grad(grad(f))`, `hessian` and `hvp` from
 * `mdml-engine/autodiff`, then Newton's method against gradient descent on the Rosenbrock function. Both runs are
 * traced, and the figure draws their paths from the tapes, with the local quadratic model (the Hessian's ellipse) at
 * two points.
 */
import { grad, hessian, hvp } from 'mdml-engine/autodiff'
import { minimize, newton, sgd } from 'mdml-engine/optim'
import { add, mul, square, sub, take, tensor, toArray, toFlat, type Value } from 'mdml-engine/tensor'
import { policies, type Run, recorder } from 'mdml-engine/trace'
import { cartesian, lines } from 'mdml-render/charts'
import { optimisationPaths, parameterPath } from 'mdml-render/trace'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'Second order: grad of grad, Hessians and Newton',
  summary: 'grad(grad(f)), hessian and hvp; Newton reaches the minimum in 12 steps where gradient descent needs 2662.',
  readme: true,
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
    { at: [-1.2, 1] as [number, number], hessian: at([-1.2, 1]), name: 'curvature at the start' },
    { at: [1, 1] as [number, number], hessian: at([1, 1]), name: 'curvature at the minimum' },
  ]
  return { result: { tapes, curvature }, tapes, notes }
}

/** The level set of the local quadratic model, d' H d / 2 = level: an ellipse along the Hessian's axes. */
function ellipse(p: [number, number], h: number[][], level: number) {
  const [a = 0, b = 0] = h[0] ?? []
  const d = h[1]?.[1] ?? 0
  const mid = (a + d) / 2
  const spread = Math.sqrt(((a - d) / 2) ** 2 + b * b)
  const angle = 0.5 * Math.atan2(2 * b, a - d)
  const [u, v] = [Math.sqrt((2 * level) / (mid + spread)), Math.sqrt((2 * level) / (mid - spread))]
  const t = Array.from({ length: 61 }, (_, i) => (i / 60) * 2 * Math.PI)
  return {
    x: t.map((k) => p[0] + u * Math.cos(k) * Math.cos(angle) - v * Math.sin(k) * Math.sin(angle)),
    y: t.map((k) => p[1] + u * Math.cos(k) * Math.sin(angle) + v * Math.sin(k) * Math.cos(angle)),
  }
}

export function View({ result }: { result: Result }) {
  const steps = parameterPath(result.tapes['newton'] as Run, 'x')
  const paths = [
    { name: `Newton (${steps.length - 1} steps)`, points: steps },
    {
      name: 'gradient descent (first 60 steps)',
      points: parameterPath(result.tapes['gradient-descent'] as Run, 'params.x'),
    },
  ]
  const colours = ['#d95f02', '#1b9e77']
  const layers = [
    ...optimisationPaths((x, y) => (1 - x) ** 2 + 10 * (y - x * x) ** 2, paths, {
      x: [-1.6, 1.4],
      y: [-0.6, 1.8],
      n: 60,
      levels: 14,
    }),
    lines(
      result.curvature.map((c, i) => ({
        ...ellipse(c.at, c.hessian, 0.3),
        name: c.name,
        colour: colours[i] ?? '#333333',
        width: 2,
      })),
    ),
  ]
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
