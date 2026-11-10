// biome-ignore-all lint/suspicious/noConsole: a measurement script that prints its results
/**
 * Re-measure the three timings quoted in docs/probprog/inventory.md section 6 (eight schools NUTS, logistic regression
 * NUTS, SVI on both): three runs each, wall-clock milliseconds, Node only. Not part of any package and not a gate.
 *
 * Reproduce, from the repository root (a run takes about a minute; nice it so the machine stays usable):
 *
 *   nice -n 15 node --conditions=source --experimental-transform-types --no-warnings docs/probprog/timings.ts
 *
 * Wall-clock time is read with `performance.now()` here, which is fine in a script outside `packages/`
 * (scripts/banned.ts only covers published source). The numbers depend on the machine: compare ratios, not seconds.
 */
import { bernoulli, halfCauchy, normal } from '../../packages/engine/src/lib/dists/index.ts'
import {
  autoMultivariateNormal,
  autoNormal,
  mcmc,
  observe,
  plate,
  sample,
  svi,
} from '../../packages/engine/src/lib/probprog/index.ts'
import { normal as drawNormal, key } from '../../packages/engine/src/lib/random/index.ts'
import { add, matmul, tensor, toFlat } from '../../packages/engine/src/lib/tensor/index.ts'

// Rubin's eight schools (the classic data).
const Y = [28, 8, -3, 7, -1, 1, 18, 12]
const SIGMA = [15, 10, 16, 11, 9, 11, 10, 18]

// Centred: 10 latent dimensions (mu, tau, theta[8]).
const schools = (): void => {
  const mu = sample('mu', normal(0, 5))
  const tau = sample('tau', halfCauchy(5))
  plate('J', 8, () => {
    const theta = sample('theta', normal(mu, tau))
    observe('y', normal(theta, tensor(SIGMA)), Y)
  })
}

// Logistic regression, n = 500 rows, d = 10 features, a fixed synthetic data set.
const N = 500
const D = 10
const X = Array.from({ length: N }, (_, i) =>
  Array.from({ length: D }, (_, j) => toFlat(drawNormal(key(1000 + i * D + j), []))[0] ?? 0),
)
const BETA = Array.from({ length: D }, (_, j) => (j % 3 === 0 ? 1 : j % 3 === 1 ? -0.5 : 0.25))
const Yb = X.map((row, i) => {
  const z = row.reduce((s, v, j) => s + v * (BETA[j] ?? 0), 0)
  const u = toFlat(drawNormal(key(5000 + i), []))[0] ?? 0
  return z + 0.5 * u > 0 ? 1 : 0
})
const logistic = (): void => {
  const alpha = sample('alpha', normal(0, 1))
  const beta = plate('features', D, () => sample('beta', normal(0, 1)))
  const logits = add(alpha, matmul(tensor(X), beta))
  plate('data', N, () => observe('y', bernoulli({ logits }), Yb))
}

function time<T>(label: string, f: () => T): void {
  const runs: number[] = []
  for (let r = 0; r < 3; r++) {
    const t0 = performance.now()
    f()
    runs.push(performance.now() - t0)
  }
  const mean = runs.reduce((a, b) => a + b, 0) / runs.length
  console.log(
    `${label.padEnd(46)} ${runs.map((v) => (v / 1000).toFixed(2)).join(' ')}  mean ${(mean / 1000).toFixed(2)} s`,
  )
}

console.log('seconds, three runs')
time('eight schools, NUTS 2 chains x (300 + 300)', () =>
  mcmc(schools, undefined, { key: key(0), numChains: 2, numWarmup: 300, numDraws: 300 }),
)
time('eight schools, SVI 1000 steps (auto normal)', () =>
  svi(schools, undefined, { key: key(0), numSteps: 1000, guide: autoNormal(schools) }),
)
time('logistic n=500 d=10, NUTS 1 chain x (200 + 200)', () =>
  mcmc(logistic, undefined, { key: key(0), numChains: 1, numWarmup: 200, numDraws: 200 }),
)
time('logistic n=500 d=10, SVI 500 steps (auto normal)', () =>
  svi(logistic, undefined, { key: key(0), numSteps: 500, guide: autoNormal(logistic) }),
)
time('logistic n=500 d=10, SVI 500 steps (auto MVN)', () =>
  svi(logistic, undefined, { key: key(0), numSteps: 500, guide: autoMultivariateNormal(logistic) }),
)
