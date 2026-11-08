/**
 * Example "A Gaussian process, tuned by a gradient": the library's negative log marginal likelihood differentiated
 * by `grad` with respect to the log of the kernel's hyperparameters, and `gpRegression` fitting them step by step on that
 * gradient. The figure has two panels from the session's `model/posterior` record: the posterior given the data with
 * the kernel's initial hyperparameters, before any learning, and with the learned ones.
 */
import { grad } from 'omai-engine/autodiff'
import { generate } from 'omai-engine/data'
import { sineCurve } from 'omai-engine/datasets'
import type { Exposition } from 'omai-engine/exposition'
import { gpRegression, negativeLogMarginalLikelihood } from 'omai-engine/gp'
import { constantKernel, productKernel, rbfKernel } from 'omai-engine/kernels'
import { key } from 'omai-engine/random'
import { tensor, toFlat } from 'omai-engine/tensor'
import { stillOf } from 'omai-render/views'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'A Gaussian process, tuned by a gradient',
  summary:
    'grad of the marginal likelihood written as a plain function, and a Gaussian process fitted on that gradient.',
  readme: false,
  alt: 'A Gaussian process on noisy sine data, before and after its hyperparameters are learned, each with its uncertainty band',
}

interface Result {
  readonly initial: Exposition
  readonly learned: Exposition
  readonly initialTitle: string
  readonly learnedTitle: string
}

export function run(): Outcome<Result> {
  // The gradient of the library's negative log marginal likelihood with respect to the kernel's two hyperparameters
  // (their logs: the amplitude and the length scale), by `grad`, checked against a finite difference.
  const x = tensor([[0], [0.7], [1.5], [2.2]])
  const y = tensor([0.1, 0.9, 0.8, -0.2])
  const kernel = productKernel(constantKernel({ value: 1 }), rbfKernel({ lengthScale: 1 }))
  const nll = negativeLogMarginalLikelihood(kernel, x, y, { alpha: 0.1 })
  const theta = [0, 0] // log amplitude 0 and log length scale 0: amplitude 1, length scale 1
  const g = Array.from(toFlat(grad(nll)(tensor(theta))))
  const h = 1e-5
  const numeric = theta.map((_, i) => {
    const up = theta.map((v, j) => (j === i ? v + h : v))
    const down = theta.map((v, j) => (j === i ? v - h : v))
    return (Number(toFlat(nll(tensor(up)))[0]) - Number(toFlat(nll(tensor(down)))[0])) / (2 * h)
  })

  // The fit: L-BFGS-B on that gradient, one step at a time.
  const { data } = generate(sineCurve, { knobs: { n: 14, noise: 0.15, design: 'sorted' }, key: key(0) })
  const fitKernel = productKernel(constantKernel({ value: 1 }), rbfKernel({ lengthScale: 0.5 }))
  const fit = gpRegression.with({ kernel: fitKernel, alpha: 0.03 }).session(data, { key: key(1) })
  fit.advance()
  const describe = (t: number): string => {
    const params = fit.fittedAt(t).params
    const [amplitude, lengthScale] = (params.theta as number[]).map((v) => Math.exp(v))
    return `amplitude ${amplitude?.toFixed(2)}, length scale ${lengthScale?.toFixed(2)}, log marginal likelihood ${params.logMarginalLikelihood.toFixed(2)}`
  }
  const look = { resolution: 120, samples: 3, seed: 3 }
  const initial = fit.probe(0, 'model/posterior', look) as Exposition
  const learned = fit.probe(fit.t, 'model/posterior', look) as Exposition
  const notes = [
    `nll at theta = [0, 0]: ${Number(toFlat(nll(tensor(theta)))[0])}`,
    `grad(nll)(tensor([0, 0])) = ${g}; finite difference ${numeric}`,
    `before learning: ${describe(0)}`,
    `after ${fit.t} steps: ${describe(fit.t)}`,
  ]
  return {
    result: {
      initial,
      learned,
      initialTitle: `initial hyperparameters: ${describe(0)}`,
      learnedTitle: `learned in ${fit.t} steps: ${describe(fit.t)}`,
    },
    tapes: { session: fit },
    notes,
  }
}

export function View({ result }: { result: Result }) {
  return (
    <Figure width={1160} height={436} title={meta.alt}>
      <Panel
        x={0}
        width={580}
        height={400}
        title={result.initialTitle}
        still={stillOf(result.initial, { mode: 'light' })}
      />
      <Panel
        x={580}
        width={580}
        height={400}
        title={result.learnedTitle}
        still={stillOf(result.learned, { mode: 'light' })}
      />
    </Figure>
  )
}
