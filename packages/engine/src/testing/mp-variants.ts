// What the tests of the message-passing variants (power, alpha, stochastic, region graphs, AMP) share: the contract
// checks every algorithm of a variant must pass (a serialised run resumes to the same state, a run sliced by a budget
// reaches the same state), and the Gaussian-latent probit regression slice they are compared on. Node only; not
// published.
import { expect } from 'vitest'
import { type Algorithm, runAlgorithm, runFor } from '../lib/trace/index.ts'

/**
 * A run stopped after `first` steps, serialised through JSON and resumed ends in exactly the state of the run that was
 * never stopped.
 */
export function expectResumable<S>(alg: Algorithm<S>, first = 3, maxSteps = 5000): void {
  const whole = runAlgorithm(alg, { maxSteps })
  const part = runFor(alg, { steps: first }, { maxSteps })
  const json = JSON.parse(JSON.stringify(alg.serialise?.(part.state))) as never
  const restored = alg.deserialise?.(json) as S
  const rest = runFor(alg, { steps: maxSteps }, { maxSteps, from: { state: restored, t: part.t } })
  expect(JSON.stringify(alg.serialise?.(rest.state))).toBe(JSON.stringify(alg.serialise?.(whole.state)))
}

/** The rows and labels of a small probit regression: an intercept and two covariates, labels from fixed weights. */
export function probitRows(n: number): { x: number[][]; y: number[] } {
  const x: number[][] = []
  const y: number[] = []
  for (let i = 0; i < n; i++) {
    // A deterministic, well spread design (no random numbers: the rows are a function of the index).
    const a = Math.sin(1.7 * i + 0.3) * 1.4
    const b = Math.cos(2.3 * i + 0.9) * 1.2
    x.push([1, a, b])
    const t = 0.3 + 1.2 * a - 0.8 * b
    // Labels follow the sign of the linear predictor with every fifth one flipped, so the posterior is proper and not extreme.
    y.push((t > 0 ? 1 : 0) ^ (i % 5 === 4 ? 1 : 0))
  }
  return { x, y }
}
