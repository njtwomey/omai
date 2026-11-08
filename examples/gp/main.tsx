/**
 * README example "A Gaussian process, tuned by a gradient": the gradient of the negative log marginal likelihood written
 * as an ordinary function and differentiated by `grad`, then `gpRegression` fitting the kernel step by step on that
 * gradient. The figure is the session's `model/posterior` record: the mean, the band and three draws.
 */
import { grad } from 'mdml-engine/autodiff'
import { generate } from 'mdml-engine/data'
import { sineCurve } from 'mdml-engine/datasets'
import type { Exposition } from 'mdml-engine/exposition'
import { gpRegression } from 'mdml-engine/gp'
import { constantKernel, gram, productKernel, rbfKernel, withTheta } from 'mdml-engine/kernels'
import { cholesky, choSolve } from 'mdml-engine/numerics/linalg'
import { key } from 'mdml-engine/random'
import { add, diagonal, eye, log, mul, sum, tensor, toFlat, type Value } from 'mdml-engine/tensor'
import { stillOf } from 'mdml-render/views'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'A Gaussian process, tuned by a gradient',
  summary:
    'grad of the marginal likelihood written as a plain function, and a Gaussian process fitted on that gradient.',
  readme: true,
  alt: 'A Gaussian process fitted to noisy sine data, with its uncertainty band',
}

export function run(): Outcome<Exposition> {
  // The gradient, as an ordinary function of the log hyperparameter.
  const x = tensor([[0], [0.7], [1.5], [2.2]])
  const y = tensor([0.1, 0.9, 0.8, -0.2])
  const kernel = rbfKernel({ lengthScale: 1 })
  const nll = (theta: Value): Value => {
    // negative log marginal likelihood, up to a constant
    const L = cholesky(add(gram(withTheta(kernel, theta), x), mul(eye(4), 0.1)))
    const a = choSolve(L, y)
    return add(mul(sum(mul(y, a)), 0.5), sum(log(diagonal(L))))
  }

  // The fit: L-BFGS-B on that gradient, one step at a time.
  const { data } = generate(sineCurve, { knobs: { n: 14, noise: 0.15, design: 'sorted' }, key: key(0) })
  const fitKernel = productKernel(constantKernel({ value: 1 }), rbfKernel({ lengthScale: 0.5 }))
  const fit = gpRegression.with({ kernel: fitKernel, alpha: 0.03 }).session(data, { key: key(1) })
  fit.advance()
  const fitted = fit.fittedAt(fit.t)
  const theta = (fitted.params.theta as number[]).map((t) => Math.exp(t).toFixed(3))
  const posterior = fit.probe(fit.t, 'model/posterior', { resolution: 120, samples: 3, seed: 3 }) as Exposition
  const notes = [
    `grad(nll)(tensor([0])) = ${toFlat(grad(nll)(tensor([0])))}`,
    `${fit.t} steps, log marginal likelihood ${fitted.params.logMarginalLikelihood.toFixed(3)}, amplitude and length scale ${theta}`,
  ]
  return { result: posterior, tapes: { session: fit }, notes }
}

export function View({ result }: { result: Exposition }) {
  return (
    <Figure width={640} height={380} title={meta.alt}>
      <Panel width={640} height={380} still={stillOf(result, { mode: 'light' })} />
    </Figure>
  )
}
