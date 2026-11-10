// biome-ignore-all lint/suspicious/noConsole: a measurement script that prints its results
/**
 * Measure how much the closure-interpretive evaluation of a bound spec costs, to decide whether a compiled evaluation plan
 * (a second evaluator of the log joint) is worth building (docs/probprog/PLAN.md section 5.8 and WP9). For the logistic,
 * radon and eight-schools benchmark models it times, per call of the log density and of its value and gradient:
 *
 * - the interpretive path: `potentialEnergy(toProgram(bound))`, which every engine runs today;
 * - a "plan ceiling": the same density written by hand with the library's own distributions and tensor ops under
 *   `target` (reverse-mode autodiff), with none of the spec evaluation, handler stack or trace; a plan that stays on the
 *   library's tensor ops and autodiff cannot be faster than this;
 * - a hand-written TypeScript density with its gradient on plain numbers, for scale (each returns the log density plus one
 *   gradient entry so the optimiser cannot drop the work).
 *
 * Reproduce, from the repository root (about a minute; nice it so the machine stays usable):
 *
 *   nice -n 15 node --conditions=source --experimental-transform-types --no-warnings docs/probprog/eval-timings.ts
 *
 * Wall-clock time is read with `performance.now()` here, which is fine in a script outside `packages/`
 * (scripts/banned.ts only covers published source). Node only, one machine: compare the ratios, not the microseconds.
 */
import * as dists from '../../packages/engine/src/lib/dists/index.ts'
import { potentialEnergy } from '../../packages/engine/src/lib/probprog/index.ts'
import { toProgram } from '../../packages/engine/src/lib/probprog/model/index.ts'
import {
  eightSchoolsCentred,
  logisticRegression,
  radon,
} from '../../packages/engine/src/lib/probprog/models/index.ts'
import { normal as drawNormal, key } from '../../packages/engine/src/lib/random/index.ts'
import { target } from '../../packages/engine/src/lib/sampling/index.ts'
import * as T from '../../packages/engine/src/lib/tensor/index.ts'

const LOG2PI = Math.log(2 * Math.PI)
const lnorm = (x: number, m: number, s: number): number => -0.5 * LOG2PI - Math.log(s) - 0.5 * ((x - m) / s) ** 2

/** Median over five blocks of `reps` calls, in microseconds per call. */
function time(f: () => unknown, reps: number): number {
  for (let i = 0; i < 30; i++) f()
  const runs: number[] = []
  for (let r = 0; r < 5; r++) {
    const t0 = performance.now()
    for (let i = 0; i < reps; i++) f()
    runs.push(((performance.now() - t0) / reps) * 1000)
  }
  runs.sort((a, b) => a - b)
  return runs[2] as number
}

interface Row {
  readonly model: string
  readonly interpretiveValue: number
  readonly interpretiveGrad: number
  readonly ceilingGrad: number
  readonly handGrad: number
}
const rows: Row[] = []

function report(model: string, interpretive: ReturnType<typeof potentialEnergy>['logDensity'], ceiling: ReturnType<typeof target>, hand: (u: Float64Array) => unknown, u: Float64Array, reps: number): void {
  const a = interpretive.logDensity(u)
  const b = ceiling.logDensity(u)
  // The ceiling is the same density: both sides carry the same constants.
  if (Math.abs(a - b) > 1e-8 * (1 + Math.abs(a))) throw new Error(`${model}: the ceiling is a different density (${a} vs ${b})`)
  rows.push({
    model,
    interpretiveValue: time(() => interpretive.logDensity(u), reps),
    interpretiveGrad: time(() => interpretive.valueAndGrad(u), reps),
    ceilingGrad: time(() => ceiling.valueAndGrad(u), reps),
    handGrad: time(() => hand(u), reps * 50),
  })
}

// Logistic regression, 50 rows and 3 features (the benchmark's own data).
{
  const d = logisticRegression.data.data as { X: number[][]; y: number[] }
  const um = potentialEnergy(toProgram(logisticRegression.bound()), undefined)
  const X = T.tensor(d.X)
  const y = T.tensor(d.y)
  const ceiling = target((w) => {
    const z = T.matmul(X, w)
    const bern = dists.bernoulli({ logits: z })
    return T.add(T.sum(dists.normal(0, 2).logProb(w)), T.sum(bern.logProb(y)))
  }, 3)
  const hand = (w: Float64Array): number => {
    let lp = 0
    const g = new Float64Array(3)
    for (let k = 0; k < 3; k++) {
      lp += lnorm(w[k] as number, 0, 2)
      g[k] = -(w[k] as number) / 4
    }
    for (let i = 0; i < d.X.length; i++) {
      const row = d.X[i] as number[]
      let z = 0
      for (let k = 0; k < 3; k++) z += (row[k] as number) * (w[k] as number)
      const yi = d.y[i] as number
      lp += yi * z - (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)))
      const p = 1 / (1 + Math.exp(-z))
      for (let k = 0; k < 3; k++) g[k] = (g[k] as number) + (yi - p) * (row[k] as number)
    }
    return lp
  }
  report('logistic regression (50 rows, 3 features)', um.logDensity, ceiling, hand, new Float64Array(3).fill(0.1), 200)
}

// Eight schools, centred: mu, log tau and eight effects.
{
  const d = eightSchoolsCentred.data.data as { y: number[]; sigma: number[] }
  const um = potentialEnergy(toProgram(eightSchoolsCentred.bound()), undefined)
  const y = T.tensor(d.y)
  const sg = T.tensor(d.sigma)
  const ceiling = target((u) => {
    const mu = T.sliceAxis(u, 0, 0, 1)
    const lt = T.sliceAxis(u, 0, 1, 2)
    const th = T.sliceAxis(u, 0, 2, 10)
    const tau = T.exp(lt)
    let lp = T.sum(dists.normal(0, 5).logProb(mu))
    lp = T.add(lp, T.sum(dists.halfCauchy(5).logProb(tau)))
    lp = T.add(lp, T.sum(lt))
    lp = T.add(lp, T.sum(dists.normal(T.reshape(mu, []), T.reshape(tau, [])).logProb(th)))
    return T.add(lp, T.sum(dists.normal(th, sg).logProb(y)))
  }, 10)
  const hand = (u: Float64Array): number => {
    const mu = u[0] as number
    const lt = u[1] as number
    const t0 = Math.exp(lt)
    const tau = Math.min(Math.max(t0, 1e-6), 1e6)
    let lp = lnorm(mu, 0, 5) + Math.log(2 / (Math.PI * 5)) - Math.log1p((t0 / 5) ** 2) + lt
    const g = new Float64Array(10)
    g[0] = -mu / 25
    g[1] = 1 - (2 * (t0 / 5) ** 2) / (1 + (t0 / 5) ** 2)
    for (let j = 0; j < 8; j++) {
      const th = u[2 + j] as number
      lp += lnorm(th, mu, tau) + lnorm(d.y[j] as number, th, d.sigma[j] as number)
      const r = (th - mu) / (tau * tau)
      g[2 + j] = -r - (th - (d.y[j] as number)) / (d.sigma[j] as number) ** 2
      g[0] = (g[0] as number) + r
      g[1] = (g[1] as number) - 1 + ((th - mu) / tau) ** 2
    }
    return lp + g[0]
  }
  report('eight schools, centred (10 coordinates)', um.logDensity, ceiling, hand, new Float64Array(10).map((_, i) => 0.1 * (i + 1) - 0.3), 150)
}

// Radon: 6 counties of 8 houses, mu, beta, log tau, log sigma and six intercepts.
{
  const d = radon.data.data as { y: number[][]; floor: number[][] }
  const um = potentialEnergy(toProgram(radon.bound()), undefined)
  const y = T.tensor(d.y)
  const fl = T.tensor(d.floor)
  const ceiling = target((u) => {
    const mu = T.sliceAxis(u, 0, 0, 1)
    const beta = T.sliceAxis(u, 0, 1, 2)
    const lt = T.sliceAxis(u, 0, 2, 3)
    const ls = T.sliceAxis(u, 0, 3, 4)
    const alpha = T.sliceAxis(u, 0, 4, 10)
    const tau = T.exp(lt)
    const sig = T.exp(ls)
    let lp = T.sum(dists.normal(0, 5).logProb(mu))
    lp = T.add(lp, T.sum(dists.normal(0, 2).logProb(beta)))
    lp = T.add(lp, T.sum(dists.halfNormal(1).logProb(tau)))
    lp = T.add(lp, T.sum(dists.halfNormal(1).logProb(sig)))
    lp = T.add(lp, T.add(T.sum(lt), T.sum(ls)))
    lp = T.add(lp, T.sum(dists.normal(T.reshape(mu, []), T.reshape(tau, [])).logProb(alpha)))
    const loc = T.add(T.reshape(alpha, [6, 1]), T.mul(T.reshape(beta, []), fl))
    return T.add(lp, T.sum(dists.normal(loc, T.reshape(sig, [])).logProb(y)))
  }, 10)
  const hn = (x: number, s: number): number => Math.log(2 / (s * Math.sqrt(2 * Math.PI))) - (x * x) / (2 * s * s)
  const hand = (u: Float64Array): number => {
    const [mu, beta, lt, ls] = [u[0] as number, u[1] as number, u[2] as number, u[3] as number]
    const tau = Math.exp(lt)
    const sig = Math.exp(ls)
    const g = new Float64Array(10)
    let lp = lnorm(mu, 0, 5) + lnorm(beta, 0, 2) + hn(tau, 1) + lt + hn(sig, 1) + ls
    g[0] = -mu / 25
    g[1] = -beta / 4
    g[2] = 1 - tau * tau
    g[3] = 1 - sig * sig
    for (let j = 0; j < 6; j++) {
      const a = u[4 + j] as number
      lp += lnorm(a, mu, tau)
      const ra = (a - mu) / (tau * tau)
      g[4 + j] = -ra
      g[0] = (g[0] as number) + ra
      g[2] = (g[2] as number) - 1 + ((a - mu) / tau) ** 2
      for (let i = 0; i < 8; i++) {
        const f = (d.floor[j] as number[])[i] as number
        const r = ((d.y[j] as number[])[i] as number) - a - beta * f
        lp += -0.5 * LOG2PI - ls - 0.5 * (r / sig) ** 2
        const rs = r / (sig * sig)
        g[4 + j] = (g[4 + j] as number) + rs
        g[1] = (g[1] as number) + rs * f
        g[3] = (g[3] as number) + (r / sig) ** 2 - 1
      }
    }
    return lp + g[0]
  }
  report('radon (6 counties x 8 houses, 10 coordinates)', um.logDensity, ceiling, hand, new Float64Array(10).map((_, i) => 0.05 * (i + 1) - 0.2), 150)
}

// The interpretive path as the rows grow: logistic regression at 50, 500 and 5000 rows (3 features).
const growth: { rows: number; interpretive: number; ceiling: number }[] = []
for (const n of [50, 500, 5000]) {
  const Xs = Array.from({ length: n }, (_, i) => Array.from({ length: 3 }, (_, j) => T.toFlat(drawNormal(key(i * 3 + j), []))[0] ?? 0))
  const ys = Array.from({ length: n }, (_, i) => (i % 3 === 0 ? 1 : 0))
  const um = potentialEnergy(toProgram(logisticRegression.definition!.bind({ X: Xs, y: ys })), undefined)
  const X = T.tensor(Xs)
  const yt = T.tensor(ys)
  const ceiling = target((w) => T.add(T.sum(dists.normal(0, 2).logProb(w)), T.sum(dists.bernoulli({ logits: T.matmul(X, w) }).logProb(yt))), 3)
  const u = new Float64Array(3).fill(0.1)
  const reps = n > 1000 ? 30 : 150
  growth.push({ rows: n, interpretive: time(() => um.logDensity.valueAndGrad(u), reps), ceiling: time(() => ceiling.valueAndGrad(u), reps) })
}

const f = (x: number): string => x.toFixed(1)
console.log('| model | interpretive log density (us) | interpretive value+grad (us) | plan ceiling value+grad (us) | hand-written value+grad (us) | interpretive / ceiling | interpretive / hand |')
console.log('| --- | --- | --- | --- | --- | --- | --- |')
for (const r of rows)
  console.log(`| ${r.model} | ${f(r.interpretiveValue)} | ${f(r.interpretiveGrad)} | ${f(r.ceilingGrad)} | ${f(r.handGrad)} | ${(r.interpretiveGrad / r.ceilingGrad).toFixed(2)}x | ${(r.interpretiveGrad / r.handGrad).toFixed(0)}x |`)
console.log('')
console.log('| logistic rows | interpretive value+grad (us) | plan ceiling value+grad (us) | interpretive / ceiling |')
console.log('| --- | --- | --- | --- |')
for (const g of growth) console.log(`| ${g.rows} | ${f(g.interpretive)} | ${f(g.ceiling)} | ${(g.interpretive / g.ceiling).toFixed(2)}x |`)
