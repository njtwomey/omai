import { adam, minimizer, sgd, newton, lbfgs, bfgs, conjugateGradient } from 'mdml-engine/optim'
import { add, mul, square, sub, take, tensor, type Value } from 'mdml-engine/tensor'
import { policies, recorder, runAlgorithm, tracePolicy, metricSeries, tensorSeries, type Run } from 'mdml-engine/trace'
const rosenbrock = (p: Value): Value => add(square(sub(1, take(p, 0))), mul(10, square(sub(take(p, 1), square(take(p, 0))))))
const first = { gd: sgd({ learningRate: 0.01 }), momentum: sgd({ learningRate: 0.005, momentum: 0.9 }), adam: adam({ learningRate: 0.1 }) }
const pol = tracePolicy({ tensors: true, roles: ['param'], detail: 'full' })
for (const [name, o] of Object.entries(first)) {
  const tape = recorder()
  const r = runAlgorithm(minimizer((p) => rosenbrock(p as Value), tensor([-1.2, 1]), o), { seed: 0, maxSteps: 20000, tracer: tape, policy: tracePolicy({}) })
  const run = tape.run()
  const names = new Set(run.events.map((e: any) => e.type + ':' + (e.name ?? '')))
  const loss = metricSeries(run, 'loss').values.map((v: number) => ({ value: v }))
  console.log(name, [...names].join(' '), loss.length, loss[0], 'first<1e-6', loss.findIndex((p: any) => p.value < 1e-6), r.status, r.steps)
}
