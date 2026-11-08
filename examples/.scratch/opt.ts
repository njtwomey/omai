import { newton, lbfgs, bfgs, conjugateGradient, minimize, sgd, adam } from 'mdml-engine/optim'
import { add, mul, square, sub, take, tensor, type Value } from 'mdml-engine/tensor'
import { policies, recorder, tensorSeries, metricSeries, type Run } from 'mdml-engine/trace'
const rosenbrock = (p: Value): Value => add(square(sub(1, take(p, 0))), mul(10, square(sub(take(p, 1), square(take(p, 0))))))
for (const [name, f] of Object.entries({ newton, lbfgs, bfgs, conjugateGradient })) {
  const tape = recorder()
  const r = (f as any)(rosenbrock, [-1.2, 1], { tracer: tape, policy: policies.teaching })
  const run: Run = tape.run()
  const kinds = new Set(run.events.map((e: any) => e.type + ':' + (e.name ?? '')))
  console.log(name, r.steps, r.status, r.fun, [...kinds].join(' '))
}
