/**
 * Operator calls that no atom, model or program of the suite makes at the shape the harness wants measured, built by hand from the
 * inputs the operators' brute-force tests use (`message-passing/operators/**\/*.test.ts`: the gate's two Gaussian branches and its zero
 * inflation, a Wishart precision with Gaussian cavities, the correlation group's cavities, a pair row on a Gaussian vector, a
 * multinomial site): the gate (made fresh by the registry, so it cannot be timed inside a run), `mvn-wishart` at dimensions 3 and 4,
 * `mvn-sd-corr-group`, `likelihood-pair-row` and `multiclass-site` with four classes. The numbers are made with the tensor library.
 */
import {
  type Arg,
  beta,
  betaEta,
  discrete,
  discreteEta,
  gamma,
  gammaEta,
  gaussian,
  gaussianEta,
  gaussianMv,
  gaussianMvEta,
  messageArg,
  type OperatorContext,
  valueArg,
  wishart,
  wishartEta,
} from '../../packages/engine/src/lib/message-passing/index.ts'
import { add, eye, linspace, mul, toFlat } from '../../packages/engine/src/lib/tensor/index.ts'

/** One call: the operator, the hook and the context. */
export interface SyntheticCall {
  readonly op: string
  readonly hook: 'ep' | 'logEvidence'
  readonly ctx: OperatorContext
}

const context = (to: number, args: readonly Arg[], params: OperatorContext['params'] = {}): OperatorContext => ({
  mode: 'ep',
  to,
  args,
  params,
  power: 1,
})

/** A mean vector and a dense positive definite covariance (row-major) of dimension `d`. */
function cavityOf(d: number, shift: number): { readonly mean: number[]; readonly covariance: number[] } {
  return {
    mean: toFlat(linspace(0.9 + shift, -0.4 + shift, d)),
    covariance: toFlat(add(mul(eye(d), 0.4), 0.1)),
  }
}

/**
 * The calls.
 *
 * @returns The synthetic calls.
 */
export function syntheticCalls(): SyntheticCall[] {
  const calls: SyntheticCall[] = []
  // The gate: two Gaussian branches (a mixture of two regressions with their own random means), and a zero inflation.
  const two = {
    arity: 2,
    branches: [
      { operator: 'gaussian', args: [{ const: 0.7 }, 0], params: { precision: 4 } },
      { operator: 'gaussian', args: [{ const: 0.7 }, 1], params: { precision: 1 } },
    ],
  }
  const gateArgs = [
    messageArg(discrete(2), discreteEta([0.3, 0.7])),
    messageArg(gaussian, gaussianEta(0.2, 0.9)),
    messageArg(gaussian, gaussianEta(1.5, 1.4)),
  ]
  for (const to of [0, 1]) calls.push({ op: 'gate', hook: 'ep', ctx: context(to, gateArgs, two) })
  const inflation = {
    arity: 1,
    branches: [{}, { operator: 'gaussian', args: [{ const: 3 }, 0], params: { precision: 2 } }],
  }
  calls.push({
    op: 'gate',
    hook: 'ep',
    ctx: context(
      1,
      [messageArg(discrete(2), discreteEta([0.4, 0.6])), messageArg(gaussian, gaussianEta(0, 1))],
      inflation,
    ),
  })
  // mvn-wishart at dimensions 2, 3 and 4: the message to the precision matrix and to the vector.
  for (const d of [2, 3, 4]) {
    const { mean, covariance } = cavityOf(d, 0)
    const zero = toFlat(mul(linspace(0, 0, d), 1))
    const w = wishart(d)
    const cavity = wishartEta(d + 4, toFlat(add(mul(eye(d), 0.6), 0.1)), d)
    const args = [
      messageArg(gaussianMv(d), gaussianMvEta(mean, covariance)),
      valueArg(gaussianMv(d), zero),
      messageArg(w, cavity),
    ]
    for (const to of [0, 2]) calls.push({ op: 'mvn-wishart', hook: 'ep', ctx: context(to, args) })
    calls.push({ op: 'mvn-wishart', hook: 'logEvidence', ctx: context(0, args) })
  }
  // mvn-sd-corr-group with three vectors: the message to a vector, to a precision and to the correlation.
  const G = gaussianMv(2)
  const groupArgs = [
    ...[0, 0.3, -0.2].map((s) => {
      const { mean, covariance } = cavityOf(2, s)
      return messageArg(G, gaussianMvEta(mean, covariance))
    }),
    messageArg(G, gaussianMvEta([0.1, 0.2], [0.6, 0.1, 0.1, 0.5])),
    messageArg(gamma, gammaEta(3, 2.5)),
    messageArg(gamma, gammaEta(2.5, 2)),
    messageArg(beta, betaEta(2, 2)),
  ]
  for (const to of [0, 4, 6])
    calls.push({ op: 'mvn-sd-corr-group', hook: 'ep', ctx: context(to, groupArgs, { groups: 3 }) })
  // A pair row: two coordinates that are linear forms of one Gaussian vector, a beta density at them.
  const row = [messageArg(G, gaussianMvEta([0.6, 1.1], [0.5, 0.35, 0.35, 0.45]))]
  calls.push({
    op: 'likelihood-pair-row',
    hook: 'ep',
    ctx: context(0, row, { likelihood: 'beta-shapes', y: 0.27, matrix: toFlat(eye(2)) }),
  })
  // A multinomial site with four classes (the suite's atoms run it with eight): a logit link, the second class observed.
  const classes = [0, 1, 2, 3].map((k) => messageArg(gaussian, gaussianEta(0.3 * k - 0.4, 0.5 + 0.2 * k)))
  for (const to of [0, 1])
    calls.push({ op: 'multiclass-site', hook: 'ep', ctx: context(to, classes, { y: 1, classes: 4, link: 'logit' }) })
  return calls
}
