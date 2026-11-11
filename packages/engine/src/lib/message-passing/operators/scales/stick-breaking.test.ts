// stick-breaking: the messages to the assignment and to each break against a brute-force integral over the breaks, with the
// log evidence against the same integral, for three and four states.
import { describe, expect, it } from 'vitest'
import { messageArg } from '../../core/args.ts'
import { beta, betaEta } from '../../core/families.ts'
import { discrete, discreteEta, discreteProbs } from '../../core/families-vector.ts'
import { lgammaOf } from '../../numerics/_scalar.ts'
import { stickBreaking } from './stick-breaking.ts'

/** p(z = k | v) for the stick-breaking weights. */
function weights(v: readonly number[]): number[] {
  const out: number[] = []
  let rest = 1
  for (const x of v) {
    out.push(rest * x)
    rest *= 1 - x
  }
  out.push(rest)
  return out
}

/** The tilted E log v_j and E log (1 - v_j) of every break, and the evidence, by a midpoint rule over the cube of breaks. */
function brute(shapes: readonly (readonly [number, number])[], r: readonly number[]) {
  const sticks = shapes.length
  const n = sticks === 2 ? 1200 : 150
  const lnorm = shapes.map(([a, b]) => lgammaOf(a + b) - lgammaOf(a) - lgammaOf(b))
  const total = n ** sticks
  let z = 0
  const el = Array.from({ length: sticks }, () => [0, 0])
  const pz = new Array<number>(sticks + 1).fill(0)
  for (let flat = 0; flat < total; flat++) {
    const v: number[] = []
    let q = flat
    for (let j = 0; j < sticks; j++) {
      v.push(((q % n) + 0.5) / n)
      q = Math.floor(q / n)
    }
    let dens = 1
    v.forEach((x, j) => {
      const [a, b] = shapes[j] as [number, number]
      dens *= Math.exp((lnorm[j] as number) + (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x))
    })
    const w = weights(v)
    const f = w.reduce((s, p, k) => s + p * (r[k] as number), 0)
    z += dens * f
    pz.forEach((_, k) => {
      pz[k] = (pz[k] as number) + dens * (w[k] as number)
    })
    v.forEach((x, j) => {
      ;(el[j] as number[])[0] = ((el[j] as number[])[0] as number) + dens * f * Math.log(x)
      ;(el[j] as number[])[1] = ((el[j] as number[])[1] as number) + dens * f * Math.log(1 - x)
    })
  }
  return {
    evidence: Math.log(z / total),
    logMoments: el.map((e) => e.map((x) => x / z)),
    meanWeights: pz.map((x) => x / total),
  }
}

describe.each([
  {
    shapes: [
      [2, 3],
      [1.5, 1],
    ] as const,
    r: [0.2, 0.5, 0.3],
  },
  {
    shapes: [
      [2, 3],
      [3, 2.5],
      [2, 2],
    ] as const,
    r: [0.1, 0.2, 0.4, 0.3],
  },
])('stick-breaking with $r.length states', ({ shapes, r }) => {
  const sticks = shapes.length
  const d = discrete(sticks + 1)
  const args = [messageArg(d, discreteEta(r)), ...shapes.map(([a, b]) => messageArg(beta, betaEta(a, b)))]
  const params = { sticks }
  const ref = brute(shapes, r)
  const run = (to: number) => stickBreaking.ep?.({ mode: 'ep', to, args, params, power: 1 }).message ?? []

  it('sends the mean weights to the assignment', () => {
    const m = discreteProbs(run(0))
    for (const [k, p] of m.entries()) expect(p).toBeCloseTo(ref.meanWeights[k] as number, 4)
  })

  it('sends each break the beta that matches the tilted log moments of the cavity times the factor', () => {
    for (let j = 0; j < sticks; j++) {
      const [a, b] = shapes[j] as [number, number]
      const eta = betaEta(a, b)
      const tilted = beta.moments(run(1 + j).map((v, i) => v + (eta[i] as number)))
      expect(tilted[0]).toBeCloseTo((ref.logMoments[j] as number[])[0] as number, 3)
      expect(tilted[1]).toBeCloseTo((ref.logMoments[j] as number[])[1] as number, 3)
    }
  })

  it('has the log evidence of the integral', () => {
    // The mass of every message is carried by the cavities: the evidence is the factor's own integral.
    const own = stickBreaking.logEvidence?.({ mode: 'ep', to: 0, args, params, power: 1 }) ?? Number.NaN
    const masses = args.reduce((s, a) => s + a.family.logPartition(a.eta ?? []), 0)
    expect(own - masses).toBeCloseTo(ref.evidence, 3)
  })

  it('has variational messages that are the expected logs of the weights, and agrees with a state observed', () => {
    const vmp = stickBreaking.vmp?.({ mode: 'vmp', to: 1, args, params, power: 1 }).message ?? []
    expect(vmp[0]).toBeCloseTo(r[0] as number, 10)
    expect(vmp[1]).toBeCloseTo(
      r.slice(1).reduce((a, x) => a + x, 0),
      10,
    )
    const first = [{ family: d, eta: undefined, value: 0 }, ...args.slice(1)]
    const m = stickBreaking.ep?.({ mode: 'ep', to: 1, args: first, params, power: 1 }).message ?? []
    // z = 0 has probability v_0: the beta gets one more success.
    const [a, b] = shapes[0] as [number, number]
    const tilted = beta.moments(m.map((v, i) => v + (betaEta(a, b)[i] as number)))
    const want = beta.moments(betaEta(a + 1, b))
    expect(tilted[0]).toBeCloseTo(want[0] as number, 8)
    expect(tilted[1]).toBeCloseTo(want[1] as number, 8)
  })
})
