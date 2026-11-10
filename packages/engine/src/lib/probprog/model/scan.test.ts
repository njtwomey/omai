// The scan node: a recurrence in the spec layer that compiles to the `markov` machinery. Its log joint is the hand-written
// markov program's, its JSON round-trips, the analyses see it, the structure draws it with a self-loop on the time plate,
// and the engines that follow a chain (the Kalman smoother, the particle filter, the chain engine, NUTS) run it.
import { describe, expect, it } from 'vitest'
import { normal as normalDist } from '../../dists/index.ts'
import { item, mul, toFlat, type Value } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import { analyse } from '../analysis.ts'
import { logJoint } from '../density.ts'
import { chainMethod, epSpecMethod, gaussianMethod, nutsMethod, particleFilterMethod } from '../inference/index.ts'
import { markov, observe, sample } from '../primitives.ts'
import { analyseSpec, structureReport } from './analysis.ts'
import { defineModel } from './builder.ts'
import { toProgram } from './compile.ts'
import { fromJSON } from './definition.ts'
import * as d from './dists/index.ts'
import { toFactorGraph } from './factors.ts'
import { emissionNodes, expandScans, unrollScans } from './scan.ts'
import { toStructure } from './structure.ts'
import { ops } from './var.ts'

const PHI = 0.8
const SX = 0.5
const SY = 0.6
const S0 = SX / Math.sqrt(1 - PHI ** 2)
const Y = [0.4, 0.1, -0.3, 0.7, 1.1, 0.2]

const ar1 = defineModel('ar1', (b) => {
  const y = b.data('y', { dims: ['time'] })
  b.scan('x', {
    over: 'time',
    init: d.normal(0, S0),
    step: ({ prev }) => d.normal(prev.mul(PHI), SX),
    emit: ({ state, observe: obs }) => obs('y', d.normal(state, SY), y),
  })
})

const markovAr1 = (y: readonly number[]) => () => {
  markov(
    'time',
    y.length,
    (t, prev: Value | undefined) => {
      const x = sample(`x${t}`, normalDist(prev === undefined ? 0 : mul(prev, PHI), t === 0 ? S0 : SX))
      observe(`y${t}`, normalDist(x, SY), y[t] ?? 0)
      return x
    },
    undefined as Value | undefined,
  )
}

describe('scan compiles to a markov program', () => {
  it('has the log joint of the hand-written markov program, site for site', () => {
    const states = [0.3, -0.2, 0.5, 0.9, 0.4, -0.1]
    const spec = toProgram(ar1.bind({ y: Y }))
    const byHand = markovAr1(Y)
    const a = logJoint(spec, undefined, Object.fromEntries(states.map((v, t) => [`x[${t}]`, v])))
    const b = logJoint(byHand, undefined, Object.fromEntries(states.map((v, t) => [`x${t}`, v])))
    expect(item(a.logJoint)).toBeCloseTo(item(b.logJoint), 12)
  })

  it('names the states and the observations <name>[t], inside one markov loop of history 1', () => {
    const a = analyse(toProgram(ar1.bind({ y: Y })), undefined)
    expect(a.sites.filter((s) => s.role === 'latent').map((s) => s.name)).toEqual(Y.map((_, t) => `x[${t}]`))
    expect(a.observed).toEqual(Y.map((_, t) => `y[${t}]`))
    expect(a.markov.length).toBe(1)
    expect(a.markov[0]?.length).toBe(Y.length)
    expect(a.markov[0]?.history).toBe(1)
  })

  it('reads series over the scanned dimension at the current step, and a later node may read the stacked states', () => {
    const m = defineModel('drift', (b) => {
      const u = b.data('u', { dims: ['time'], role: 'covariate' })
      const y = b.data('y', { dims: ['time'] })
      const k = b.latent('k', d.normal(0, 1))
      const x = b.scan('x', {
        over: 'time',
        init: d.normal(0, 1),
        step: ({ prev }) => d.normal(prev.add(u.mul(k)), 0.3),
        emit: ({ state, observe: obs }) => obs('y', d.normal(state, 0.5), y),
      })
      b.derived('level', x.mean('time'))
    })
    const bound = m.bind({ u: [1, 2, 0], y: [0.1, 0.5, 0.2] })
    const lj = logJoint(toProgram(bound), undefined, { k: 0.3, 'x[0]': 0.1, 'x[1]': 0.4, 'x[2]': 0.9 })
    const x = [0.1, 0.4, 0.9]
    const u = [1, 2, 0]
    const y = [0.1, 0.5, 0.2]
    const lp = (v: number, loc: number, sd: number): number =>
      -0.5 * Math.log(2 * Math.PI) - Math.log(sd) - 0.5 * ((v - loc) / sd) ** 2
    let want = lp(0.3, 0, 1)
    x.forEach((v, t) => {
      want += t === 0 ? lp(v, 0, 1) : lp(v, (x[t - 1] as number) + (u[t] as number) * 0.3, 0.3)
      want += lp(y[t] as number, v, 0.5)
    })
    expect(item(lj.logJoint)).toBeCloseTo(want, 10)
    const recorded = lj.trace.sites.get('level')
    expect(recorded && 'value' in recorded ? item(recorded.value as Value) : Number.NaN).toBeCloseTo(
      (0.1 + 0.4 + 0.9) / 3,
      12,
    )
  })

  it('leaves out an emission whose data is unbound, so the scan runs forward for prediction', () => {
    const open = defineModel('open', (b) => {
      const y = b.data('y', { dims: ['time'], optional: true })
      b.scan('x', {
        over: 'time',
        init: d.normal(0, 1),
        step: ({ prev }) => d.normal(prev, 1),
        emit: ({ state, observe: obs }) => obs('y', d.normal(state, 1), y),
      })
    })
    const forward = open.bind({}, { sizes: { time: 3 } })
    const a = analyse(toProgram(forward, { generated: false }), undefined)
    expect(a.observed).toEqual([])
    expect(a.sites.filter((s) => s.role === 'latent').length).toBe(3)
  })
})

describe('scan in the spec layer', () => {
  it('round-trips through JSON, validated', () => {
    const back = fromJSON(JSON.parse(JSON.stringify(ar1)))
    expect(back.spec).toEqual(ar1.spec)
    const lj = (def: typeof ar1): number =>
      item(
        logJoint(toProgram(def.bind({ y: Y })), undefined, Object.fromEntries(Y.map((_, t) => [`x[${t}]`, 0.1 * t])))
          .logJoint,
      )
    expect(lj(back)).toBe(lj(ar1))
  })

  it('refuses a scan with a dimension left inside, a repeated emission name and an undeclared reference', () => {
    expect(() =>
      defineModel('bad', (b) => {
        b.range('time')
        b.range('series')
        const s = b.data('s', { dims: ['series'] })
        b.scan('x', { over: 'time', init: d.normal(s, 1), step: ({ prev }) => d.normal(prev, 1) })
      }),
    ).toThrow(/left inside the scan/)
    expect(() =>
      defineModel('bad', (b) => {
        const y = b.data('y', { dims: ['time'] })
        b.scan('x', {
          over: 'time',
          init: d.normal(0, 1),
          step: ({ prev }) => d.normal(prev, 1),
          emit: ({ state, observe: obs }) => {
            obs('y', d.normal(state, 1), y)
            obs('y', d.normal(state, 2), y)
          },
        })
      }),
    ).toThrow(/already used/)
    expect(() =>
      defineModel('bad', (b) => {
        b.scan('x', { over: 'time', init: d.normal(0, 1), step: () => d.normal(b.const(0), 1) })
        b.latent('later', d.normal(0, 1))
      }),
    ).not.toThrow()
    const tampered = JSON.parse(JSON.stringify(ar1))
    tampered.nodes[1].step.params.loc.args[0].name = 'x'
    expect(() => fromJSON(tampered)).toThrow(/not a node defined before/)
  })

  it('composes: include prefixes the scan, its emissions and its previous-state references', () => {
    const outer = defineModel('outer', (b) => {
      const y = b.data('series', { dims: ['time'] })
      b.include(ar1, { prefix: 'a', bind: { y }, dims: { time: 'time' } })
    })
    const bound = outer.bind({ series: Y })
    const names = analyse(toProgram(bound), undefined).sites.map((s) => s.name)
    expect(names).toContain('a/x[0]')
    expect(names).toContain('a/y[5]')
    const lj = logJoint(toProgram(bound), undefined, Object.fromEntries(Y.map((_, t) => [`a/x[${t}]`, 0.1 * t])))
    const plain = logJoint(
      toProgram(ar1.bind({ y: Y })),
      undefined,
      Object.fromEntries(Y.map((_, t) => [`x[${t}]`, 0.1 * t])),
    )
    expect(item(lj.logJoint)).toBeCloseTo(item(plain.logJoint), 12)
  })
})

describe('the analyses see a scan', () => {
  const a = analyseSpec(ar1)
  it('counts it as one latent with its emissions as observed nodes, and says what it is', () => {
    expect(a.latent).toEqual(['x'])
    expect(a.observed).toEqual(['y'])
    expect(a.hasLikelihood).toBe(true)
    expect(a.continuous).toEqual(['x'])
    expect(a.features).toContain('recurrences')
    expect(a.scans.map((s) => [s.name, s.over, s.family, s.stateKind, s.emissions])).toEqual([
      ['x', 'time', 'normal', 'continuous', ['y']],
    ])
    expect(emissionNodes(ar1.spec.nodes[1] as never).map((n) => n.name)).toEqual(['y'])
    expect(expandScans(ar1.spec).nodes.length).toBe(3)
  })

  it('finds the Gaussian chain: the whole model is linear-Gaussian, one block over the series', () => {
    expect(a.linearGaussian.holds).toBe(true)
    const bound = analyseSpec(ar1.bind({ y: Y }))
    expect(bound.linearGaussianBlocks).toEqual([
      expect.objectContaining({ sites: ['x'], given: [], observed: ['y'], dimension: Y.length, whole: true }),
    ])
    expect(bound.scans[0]?.steps).toBe(Y.length)
  })

  it('a coefficient that is a latent breaks the whole-model test but leaves the chain a block given it', () => {
    const m = defineModel('ar1-phi', (b) => {
      const y = b.data('y', { dims: ['time'] })
      const phi = b.latent('phi', d.normal(0.5, 0.3))
      b.scan('x', {
        over: 'time',
        init: d.normal(0, 1),
        step: ({ prev }) => d.normal(prev.mul(phi), 0.5),
        emit: ({ state, observe: obs }) => obs('y', d.normal(state, 0.6), y),
      })
    })
    const an = analyseSpec(m.bind({ y: Y }))
    expect(an.linearGaussian.holds).toBe(false)
    expect(an.linearGaussianBlocks.map((x) => [x.sites, x.given, x.whole])).toContainEqual([['x'], ['phi'], false])
    // The latent phi is used by the recurrence, which no conjugate form covers.
    expect(an.conjugacy.every((c) => !c.sites.includes('phi') || !c.holds)).toBe(true)
    expect(an.scans[0]?.parameters).toEqual(['phi'])
  })

  it('a non-linear step is not Gaussian-linear, and the structure report names the filter for it', () => {
    const sv = defineModel('nonlinear', (b) => {
      const y = b.data('y', { dims: ['time'] })
      b.scan('x', {
        over: 'time',
        init: d.normal(0, 1),
        step: ({ prev }) => d.normal(ops.tanh(prev).mul(2), 0.5),
        emit: ({ state, observe: obs }) => obs('y', d.normal(state, 0.4), y),
      })
    })
    const an = analyseSpec(sv)
    expect(an.linearGaussian.holds).toBe(false)
    expect(an.scans[0]?.linearGaussian.reason).toMatch(/not affine/)
    expect(structureReport(sv).suggestions.map((s) => [s.kind, s.use[0]?.name])).toContainEqual([
      'recurrence',
      'particle-filter',
    ])
  })

  it('no minibatch axis over the scanned dimension', () => {
    expect(structureReport(ar1.bind({ y: Y })).minibatchAxes).toEqual([])
  })
})

describe('scan drawn and unrolled', () => {
  it('draws the state and its emission in the time plate, with the recurrence as a self-loop', () => {
    const s = toStructure(ar1.bind({ y: Y }))
    expect(s.nodes.map((n) => [n.id, n.role, n.group])).toEqual([
      ['x', 'latent', 'plate:time'],
      ['y', 'observed', 'plate:time'],
    ])
    expect(s.edges.map((e) => `${e.from}>${e.to}`)).toEqual(['x>y', 'x>x'])
    expect(s.groups).toEqual([expect.objectContaining({ id: 'plate:time', count: Y.length })])
    expect(s.nodes[0]?.detail).toContain('x[t-1]')
    const f = toStructure(ar1.bind({ y: Y }), { view: 'factor' })
    expect(f.nodes.map((n) => n.id)).toEqual(['x', 'factor:x', 'y', 'factor:y'])
  })

  it('unrolls to scalar nodes with the same log joint', () => {
    const bound = ar1.bind({ y: Y })
    const flat = unrollScans(bound)
    expect(flat.spec.nodes.filter((n) => n.kind !== 'data').map((n) => n.name)).toEqual(
      Y.flatMap((_, t) => [`x[${t}]`, `y[${t}]`]),
    )
    const values = Object.fromEntries(Y.map((_, t) => [`x[${t}]`, 0.2 * t - 0.3]))
    expect(item(logJoint(toProgram(flat), undefined, values).logJoint)).toBeCloseTo(
      item(logJoint(toProgram(bound), undefined, values).logJoint),
      12,
    )
  })

  it('refuses to unroll a scan that reads a computed series or whose stacked states are read', () => {
    const m = defineModel('reads', (b) => {
      const y = b.data('y', { dims: ['time'] })
      const x = b.scan('x', {
        over: 'time',
        init: d.normal(0, 1),
        step: ({ prev }) => d.normal(prev, 1),
        emit: ({ state, observe: obs }) => obs('y', d.normal(state, 1), y),
      })
      b.derived('m', x.mean('time'))
    })
    expect(() => unrollScans(m.bind({ y: Y }))).toThrow(/stacked states/)
  })
})

describe('the factor graph of a Gaussian scan', () => {
  it('unrolls the chain and refuses what it cannot', () => {
    const g = toFactorGraph(ar1.bind({ y: Y }))
    expect(g.issues).toEqual([])
    expect(g.graph?.variables.map((v) => v.name).filter((n) => /^x\[\d+\]$/.test(n))).toEqual(
      Y.map((_, t) => `x[${t}]`),
    )
    const nonlinear = defineModel('nl', (b) => {
      const y = b.data('y', { dims: ['time'] })
      b.scan('x', {
        over: 'time',
        init: d.normal(0, 1),
        step: ({ prev }) => d.normal(ops.tanh(prev), 0.5),
        emit: ({ state, observe: obs }) => obs('y', d.normal(state, 0.4), y),
      })
    })
    const refused = toFactorGraph(nonlinear.bind({ y: Y }))
    expect(refused.graph).toBeUndefined()
    expect(refused.issues.length).toBeGreaterThan(0)
  })
})

// Kalman by hand for the AR(1) state space (a scalar filter and smoother), independent of every library engine.
function kalmanSmoother(y: readonly number[]): { mean: number[]; sd: number[]; logEvidence: number } {
  const T = y.length
  const m: number[] = []
  const P: number[] = []
  const mp: number[] = []
  const Pp: number[] = []
  let logEvidence = 0
  for (let t = 0; t < T; t++) {
    mp[t] = t === 0 ? 0 : PHI * (m[t - 1] as number)
    Pp[t] = t === 0 ? S0 * S0 : PHI * PHI * (P[t - 1] as number) + SX * SX
    const S = (Pp[t] as number) + SY * SY
    const K = (Pp[t] as number) / S
    const e = (y[t] as number) - (mp[t] as number)
    m[t] = (mp[t] as number) + K * e
    P[t] = (1 - K) * (Pp[t] as number)
    logEvidence += -0.5 * (Math.log(2 * Math.PI * S) + (e * e) / S)
  }
  const ms = [...m]
  const Ps = [...P]
  for (let t = T - 2; t >= 0; t--) {
    const G = ((P[t] as number) * PHI) / (Pp[t + 1] as number)
    ms[t] = (m[t] as number) + G * ((ms[t + 1] as number) - (mp[t + 1] as number))
    Ps[t] = (P[t] as number) + G * G * ((Ps[t + 1] as number) - (Pp[t + 1] as number))
  }
  return { mean: ms, sd: Ps.map(Math.sqrt), logEvidence }
}

describe('the engines that follow a chain run a scan', () => {
  const bound = ar1.bind({ y: Y })
  const exact = kalmanSmoother(Y)

  it('the Kalman smoother (the gaussian method) is exact, evidence included', () => {
    const m = gaussianMethod(bound)
    expect(m.check?.(analyseSpec(bound)).applies).toBe(true)
    const s = session(m, { config: { method: 'kalman' }, steps: 3, mode: 'eager' })
    const post = m.posterior?.(s.state, s.config)
    Y.forEach((_, t) => {
      expect(item(post?.mean(`x[${t}]`) as never)).toBeCloseTo(exact.mean[t] as number, 9)
      expect(item(post?.stddev(`x[${t}]`) as never)).toBeCloseTo(exact.sd[t] as number, 9)
    })
    expect(post?.logEvidence?.value).toBeCloseTo(exact.logEvidence, 9)
  })

  it('the bootstrap particle filter estimates the evidence and the filtering mean', () => {
    const m = particleFilterMethod(bound)
    expect(m.check?.(analyseSpec(bound)).applies).toBe(true)
    const s = session(m, { config: { numParticles: 4000 }, steps: 50, mode: 'eager' })
    expect(s.t).toBe(Y.length)
    const post = m.posterior?.(s.state, s.config)
    expect(Math.abs((post?.logEvidence?.value ?? Number.NaN) - exact.logEvidence)).toBeLessThan(0.15)
  })

  it('expectation propagation on the unrolled factor graph is exact on the chain (a tree of Gaussians)', () => {
    const m = epSpecMethod(bound)
    expect(m.check?.(analyseSpec(bound)).applies).toBe(true)
    const s = session(m, { steps: 300, mode: 'eager' })
    const post = m.posterior?.(s.state, s.config)
    Y.forEach((_, t) => {
      expect(item(post?.mean(`x[${t}]`) as never)).toBeCloseTo(exact.mean[t] as number, 6)
      expect(item(post?.stddev(`x[${t}]`) as never)).toBeCloseTo(exact.sd[t] as number, 6)
    })
  })

  it('NUTS samples the unrolled density and agrees with the smoother', () => {
    const m = nutsMethod(bound)
    const s = session(m, { config: { numChains: 2, numWarmup: 200, numDraws: 400 }, steps: 5000, mode: 'eager' })
    const post = m.posterior?.(s.state, s.config)
    for (const t of [0, 3, 5]) {
      expect(Math.abs(item(post?.mean(`x[${t}]`) as never) - (exact.mean[t] as number))).toBeLessThan(0.15)
      expect(Math.abs(item(post?.stddev(`x[${t}]`) as never) - (exact.sd[t] as number))).toBeLessThan(0.1)
    }
  })

  it('a non-Gaussian scan is refused by the Kalman smoother, with the reason', () => {
    const nl = defineModel('nl', (b) => {
      const y = b.data('y', { dims: ['time'] })
      b.scan('x', {
        over: 'time',
        init: d.normal(0, 1),
        step: ({ prev }) => d.normal(ops.tanh(prev), 0.5),
        emit: ({ state, observe: obs }) => obs('y', d.normal(state, 0.4), y),
      })
    })
    const verdict = gaussianMethod(nl.bind({ y: Y })).check?.(analyseSpec(nl))
    expect(verdict?.applies).toBe(false)
    expect(verdict?.reason).toMatch(/not affine/)
  })
})

describe('a hidden Markov chain as a scan', () => {
  const A = [
    [0.9, 0.1],
    [0.2, 0.8],
  ]
  const pi = [0.6, 0.4]
  const mu = [-1, 1]
  const sd = 0.7
  const yy = [-0.8, -1.2, 0.9, 1.4, 0.7]
  const hmm = defineModel('hmm', (b) => {
    const y = b.data('y', { dims: ['time'] })
    const P = b.const(A, ['from'])
    const means = b.const(mu, ['state'])
    b.scan('z', {
      over: 'time',
      init: d.categorical({ probs: pi }),
      step: ({ prev }) => d.categorical({ probs: ops.at(P, prev) }),
      emit: ({ state, observe: obs }) => obs('y', d.normal(ops.at(means, state), sd), y),
    })
  })
  const forwardBackward = (): { marginals: number[][]; logEvidence: number } => {
    const lik = (t: number): number[] =>
      mu.map((m) => Math.exp(-0.5 * (((yy[t] as number) - m) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI)))
    const alpha: number[][] = []
    const c: number[] = []
    yy.forEach((_, t) => {
      const prior =
        t === 0
          ? pi
          : [0, 1].map((j) =>
              (alpha[t - 1] as number[]).reduce((a, v, i) => a + v * ((A[i] as number[])[j] as number), 0),
            )
      const a = prior.map((p, j) => p * (lik(t)[j] as number))
      c[t] = a.reduce((x, v) => x + v, 0)
      alpha[t] = a.map((v) => v / (c[t] as number))
    })
    const beta: number[][] = Array.from({ length: yy.length }, () => [1, 1])
    for (let t = yy.length - 2; t >= 0; t--)
      beta[t] = [0, 1].map(
        (i) =>
          [0, 1].reduce(
            (a, j) =>
              a +
              ((A[i] as number[])[j] as number) * (lik(t + 1)[j] as number) * ((beta[t + 1] as number[])[j] as number),
            0,
          ) / (c[t + 1] as number),
      )
    return {
      marginals: yy.map((_, t) => (alpha[t] as number[]).map((a, j) => a * ((beta[t] as number[])[j] as number))),
      logEvidence: c.reduce((a, v) => a + Math.log(v), 0),
    }
  }

  it('the chain engine gives the forward-backward marginals and evidence', () => {
    const bound = hmm.bind({ y: yy })
    const m = chainMethod(bound)
    expect(m.check?.(analyseSpec(bound)).applies).toBe(true)
    const s = session(m, { mode: 'eager', steps: 1 })
    const post = m.posterior?.(s.state, s.config)
    const want = forwardBackward()
    yy.forEach((_, t) => {
      expect(item(post?.mean(`z[${t}]`) as never)).toBeCloseTo((want.marginals[t] as number[])[1] as number, 9)
    })
    expect(post?.logEvidence?.value).toBeCloseTo(want.logEvidence, 9)
    expect(toFlat(post?.mean('z[0]') as never).length).toBe(1)
  })

  it('the analysis calls it a hidden Markov chain and the report names the chain engine', () => {
    const a = analyseSpec(hmm.bind({ y: yy }))
    expect(a.scans[0]?.stateKind).toBe('finite-discrete')
    expect(a.finiteDiscrete).toEqual(['z'])
    expect(structureReport(hmm.bind({ y: yy })).suggestions.map((x) => [x.kind, x.use[0]?.name])).toContainEqual([
      'recurrence',
      'chain',
    ])
  })
})
