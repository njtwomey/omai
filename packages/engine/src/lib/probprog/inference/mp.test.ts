// Expectation propagation, variational message passing and belief propagation as inference methods on specs: each matches the
// closed form of a model it is exact on, puts the engine's tape on the session (messages by factor and variable id, marginals
// by variable id), reads any state as a posterior by site name, resumes, cancels and snapshots like the other methods, and says
// why it does not apply (the node and the reason, or the operator with no message rule).
import { describe, expect, it } from 'vitest'
import { expectCancelAndContinue, expectResumeEqual, expectSnapshotCadence } from '../../../testing/methods.ts'
import { probitData, probitModel } from '../../../testing/probit-slice.ts'
import { Y4 } from '../../../testing/probprog.ts'
import { toFlat } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import * as d from '../model/dists/index.ts'
import { analyseSpec, defineModel, type NamedValue, ops } from '../model/index.ts'
import { inferenceMethodFor, methodKinds, modelRegistry } from './models.ts'
import { bpSpecMethod, epSpecMethod, vmpSpecMethod } from './mp.ts'

const exactPrecision = 1 / 4 + Y4.length / 2.25
const exactMean = Y4.reduce((a, v) => a + v, 0) / 2.25 / exactPrecision
const exactSd = Math.sqrt(1 / exactPrecision)

const normalMean = defineModel('normal-mean', (b) => {
  b.observe('y', d.normal(b.latent('mu', d.normal(0, 2)), 1.5), b.data('y', { dims: ['data'] }))
})
const data = probitData()
// Twelve rows are enough for the conventions (steps, resuming, cancelling) and keep the tests quick.
const X = data.X.slice(0, 12)
const y = data.y.slice(0, 12)

describe('epSpecMethod', () => {
  it('equals the closed-form posterior of a conjugate Gaussian model, with the posterior keyed by site', () => {
    const m = epSpecMethod(normalMean.bind({ y: Y4 }))
    const s = session(m, { steps: 100, mode: 'eager' })
    expect(s.status).toBe('converged')
    const post = m.posterior?.(s.state, s.config)
    expect(post?.status).toBe('converged')
    expect(post?.sites).toEqual(['mu'])
    expect(Math.abs((toFlat(post?.mean('mu') as never)[0] as number) - exactMean)).toBeLessThan(1e-10)
    expect(Math.abs((toFlat(post?.stddev('mu') as never)[0] as number) - exactSd)).toBeLessThan(1e-10)
    expect(post?.logEvidence?.method).toBe('ep-energy')
    expect(post?.diagnostics['improper']).toEqual([])
    expect(m.family).toBe('message-passing')
    expect(m.name).toBe('ep')
  })

  it('puts the engine tape on the session: scalars per sweep, message events by factor and variable id, marginals by id', () => {
    const m = epSpecMethod(normalMean.bind({ y: Y4 }))
    const s = session(m, { steps: 50, mode: 'eager', logs: 'all' })
    const run = s.run()
    const names = new Set(run.events.flatMap((e) => (e.type === 'message' ? [e.kind] : 'name' in e ? [e.name] : [])))
    for (const p of ['mp/change', 'mp/sweeps', 'mp/updates', 'mp/evidence', 'mp/message', 'mp/marginals'])
      expect(names.has(p), `${p} in ${[...names]}`).toBe(true)
    const message = run.events.find((e) => e.type === 'message' && e.kind === 'mp/message')
    expect(message).toMatchObject({ from: expect.stringMatching(/^factor:/), to: expect.any(String) })
    const marginals = run.events.find((e) => e.type === 'message' && e.kind === 'mp/marginals')
    expect(Object.keys((marginals as { payload: object }).payload)).toEqual(['mu', 'y[0]', 'y[1]', 'y[2]', 'y[3]'])
    // The posterior of a state is also a derived path, as JSON.
    expect(s.probe(s.t, 'mp/posterior')).toMatchObject({ engine: 'ep', form: 'variational', sites: ['mu'] })
  })

  it('reads a state before any message as a posterior with no information, and a midway state as a partial one', () => {
    const m = epSpecMethod(probitModel.bind({ X, y }))
    const s = session(m, { config: { granularity: 'factor' }, steps: 5000, mode: 'live' })
    const start = m.posterior?.(s.state, s.config)
    expect(start?.diagnostics['sweeps']).toBe(0)
    expect(start?.status).toBe('stopped')
    s.advance({ steps: 5 })
    const part = m.posterior?.(s.state, s.config)
    expect(part?.diagnostics['sweeps']).toBe(0)
    s.advance({ steps: 5000 })
    expect(m.posterior?.(s.state, s.config)?.status).toBe('converged')
  })

  it('resumes from a serialised state to the same posterior, mid-sweep too', () => {
    const m = epSpecMethod(probitModel.bind({ X, y }))
    expectResumeEqual(m, { config: { tol: 1e-10 }, steps: 500, at: 2 })
    expectResumeEqual(m, { config: { granularity: 'factor', tol: 1e-10 }, steps: 800, at: 40 })
  })

  it('cancels between steps and continues, and snapshots on the cadence of a fake clock', () => {
    const m = epSpecMethod(probitModel.bind({ X, y }))
    expectCancelAndContinue(m, { config: { granularity: 'factor' }, steps: 800, cancelAfterSlices: 7 })
    const snaps = expectSnapshotCadence(m, {
      config: { granularity: 'message', tol: 1e-12 },
      steps: 100_000,
      budgetMs: 3500,
      everyMs: 1000,
      tick: 100,
    })
    expect(snaps.length).toBeGreaterThanOrEqual(3)
  })

  it('takes the knobs of its engine (schedule, damping, improper policy) and the budget knobs of every method', () => {
    const m = epSpecMethod(normalMean.bind({ y: Y4 }))
    const knobs = Object.keys(m.config.spaces)
    for (const k of [
      'schedule',
      'granularity',
      'damping',
      'improper',
      'tol',
      'maxSweeps',
      'budgetMs',
      'snapshotEveryMs',
    ])
      expect(knobs).toContain(k)
    const flooding = session(m, { config: { schedule: 'flooding', damping: 0.2 }, steps: 400, mode: 'eager' })
    expect(
      Math.abs((toFlat(m.posterior?.(flooding.state, flooding.config)?.mean('mu') as never)[0] as number) - exactMean),
    ).toBeLessThan(1e-6)
  })
})

describe('vmpSpecMethod', () => {
  it('is exact on a conjugate model and has the variational bound on its tape', () => {
    const coin = defineModel('coin', (b) => {
      b.observe(
        'y',
        d.bernoulli({ probs: b.latent('p', d.beta(2, 3)) }),
        b.data('y', { dims: ['flip'], support: 'binary' }),
      )
    })
    const m = vmpSpecMethod(coin.bind({ y: [1, 0, 1, 1, 0, 1, 1] }))
    const s = session(m, { steps: 100, mode: 'eager', logs: 'all' })
    expect(Math.abs((toFlat(m.posterior?.(s.state, s.config)?.mean('p') as never)[0] as number) - 7 / 12)).toBeLessThan(
      1e-9,
    )
    expect(m.posterior?.(s.state, s.config)?.logEvidence?.method).toBe('elbo')
    expect(s.run().events.some((e) => 'name' in e && e.name === 'mp/elbo')).toBe(true)
  })

  it('takes a Gaussian hierarchy and gives the mean-field answer: the right means, a narrower spread than EP', () => {
    const schools = defineModel('schools', (b) => {
      const j = b.range('school')
      const mu = b.latent('mu', d.normal(0, 10))
      const theta = b.latent('theta', d.normal(mu, 6), { dims: [j] })
      b.observe('y', d.normal(theta, b.data('sigma', { dims: [j] })), b.data('y', { dims: [j] }))
    })
    const bound = schools.bind({ y: [28, 8, -3, 7, -1], sigma: [15, 10, 16, 11, 9] })
    const vmp = vmpSpecMethod(bound)
    const ep = epSpecMethod(bound)
    const a = session(vmp, { config: { tol: 1e-12 }, steps: 500, mode: 'eager' })
    const e = session(ep, { config: { tol: 1e-12 }, steps: 500, mode: 'eager' })
    const pa = vmp.posterior?.(a.state, a.config)
    const pe = ep.posterior?.(e.state, e.config)
    expect(
      Math.abs((toFlat(pa?.mean('mu') as never)[0] as number) - (toFlat(pe?.mean('mu') as never)[0] as number)),
    ).toBeLessThan(1e-6)
    expect(toFlat(pa?.stddev('mu') as never)[0] as number).toBeLessThan(toFlat(pe?.stddev('mu') as never)[0] as number)
    expect(pa?.mean('theta').shape).toEqual([5])
  })

  it('takes a regression with scalar weights and a hierarchy with affine means: exact means, a bound that rises', () => {
    const xs = [-1, -0.5, 0.2, 0.8, 1.5]
    const ys = [-1.1, -0.4, 0.5, 1.0, 1.9]
    const line = defineModel('line', (b) => {
      const n = b.range('row')
      const a = b.latent('a', d.normal(0, 2))
      const s = b.latent('s', d.normal(0, 2))
      b.observe('y', d.normal(ops.add(a, ops.mul(s, b.data('x', { dims: [n] }))), 0.5), b.data('y', { dims: [n] }))
    })
    const m = vmpSpecMethod(line.bind({ x: xs, y: ys }))
    expect(m.check?.(analyseSpec(line))).toMatchObject({ applies: true })
    const run = session(m, { config: { tol: 1e-12 }, steps: 500, mode: 'eager' })
    const post = m.posterior?.(run.state, run.config)
    // The exact posterior of (a, s) is Gaussian: its mean is the mean field's.
    const tau = 4
    const n = xs.length
    const sx = xs.reduce((t, v) => t + v, 0)
    const sxx = xs.reduce((t, v) => t + v * v, 0)
    const P = [1 / 4 + tau * n, tau * sx, tau * sx, 1 / 4 + tau * sxx]
    const h = [tau * ys.reduce((t, v) => t + v, 0), tau * xs.reduce((t, v, i) => t + v * (ys[i] as number), 0)]
    const det = (P[0] as number) * (P[3] as number) - (P[1] as number) ** 2
    expect(toFlat(post?.mean('a') as never)[0]).toBeCloseTo(
      ((P[3] as number) * (h[0] as number) - (P[1] as number) * (h[1] as number)) / det,
      6,
    )
    expect(toFlat(post?.mean('s') as never)[0]).toBeCloseTo(
      ((P[0] as number) * (h[1] as number) - (P[2] as number) * (h[0] as number)) / det,
      6,
    )
    const bounds = run
      .run()
      .events.filter((e) => 'name' in e && e.name === 'mp/elbo')
      .map((e) => (e as unknown as { value: number }).value)
    expect(bounds.length).toBeGreaterThan(3)
    for (let k = 1; k < bounds.length; k++)
      expect(bounds[k] as number).toBeGreaterThanOrEqual((bounds[k - 1] as number) - 1e-9)
    // A hierarchy whose group means are affine in a covariate.
    const hier = defineModel('hier', (b) => {
      const j = b.range('group')
      const alpha = b.latent('alpha', d.normal(0, 3))
      const beta = b.latent('beta', d.normal(0, 3))
      const theta = b.latent('theta', d.normal(ops.add(alpha, ops.mul(beta, b.data('z', { dims: [j] }))), 1), {
        dims: [j],
      })
      b.observe('y', d.normal(theta, 0.8), b.data('y', { dims: [j] }))
    })
    const hv = vmpSpecMethod(hier.bind({ z: [0, 1, 2, 3], y: [0.2, 1.1, 1.8, 3.2] }))
    expect(hv.check?.(analyseSpec(hier))).toMatchObject({ applies: true })
    const hr = session(hv, { config: { tol: 1e-12 }, steps: 800, mode: 'eager' })
    expect(hr.status).toBe('converged')
    expect(toFlat(hv.posterior?.(hr.state, hr.config)?.mean('beta') as never)[0] as number).toBeGreaterThan(0.5)
  })

  it('takes a probit regression (the Albert-Chib augmentation) and raises the bound at every sweep', () => {
    const m = vmpSpecMethod(probitModel.bind({ X, y }))
    expect(m.check?.(analyseSpec(probitModel))).toMatchObject({ applies: true })
    const s = session(m, { config: { tol: 1e-10 }, steps: 300, mode: 'eager' })
    expect(s.status).toBe('converged')
    const bounds = s
      .run()
      .events.filter((e) => 'name' in e && e.name === 'mp/elbo')
      .map((e) => (e as unknown as { value: number }).value)
    expect(bounds.length).toBeGreaterThan(3)
    for (let k = 1; k < bounds.length; k++)
      expect(bounds[k] as number).toBeGreaterThanOrEqual((bounds[k - 1] as number) - 1e-9)
    // The weights point the same way as under expectation propagation, with a narrower spread.
    const ep = epSpecMethod(probitModel.bind({ X, y }))
    const e = session(ep, { config: { tol: 1e-10 }, steps: 300, mode: 'eager' })
    const pv = m.posterior?.(s.state, s.config)
    const pe = ep.posterior?.(e.state, e.config)
    const mv = toFlat(pv?.mean('w') as never)
    const me = toFlat(pe?.mean('w') as never)
    for (const [i, v] of mv.entries()) expect(Math.sign(v as number)).toBe(Math.sign(me[i] as number))
    expect(Math.abs((mv[1] as number) - (me[1] as number))).toBeLessThan(0.5)
  })

  it('refuses a hard comparison and names the operator and why it has no variational rule', () => {
    const hard = defineModel('hard', (b) => {
      const n = b.range('row')
      const Xd = b.data('X', { dims: [n, 'feature'] })
      const w = b.latent('w', d.normal(0, 1), { dims: ['feature'] })
      b.observe('y', d.bernoulli({ probs: ops.gt(Xd.matmul(w), 0) }), b.data('y', { dims: [n], support: 'binary' }))
    })
    const m = vmpSpecMethod(hard.bind({ X, y }))
    const v = m.check?.(analyseSpec(hard))
    expect(v).toMatchObject({ applies: false, missing: ['vmp message rules'] })
    expect(v?.reason).toMatch(/greater-than has no variational message rule/)
    expect(v?.hint).toMatch(/expectation propagation/)
    expect(() => session(m, { steps: 5, mode: 'eager' })).toThrow(/greater-than has no variational/)
  })
})

describe('bpSpecMethod', () => {
  const net = defineModel('net', (b) => {
    const rain = b.latent('rain', d.bernoulli({ probs: 0.3 }))
    const sprinkler = b.latent('sprinkler', d.bernoulli({ probs: 0.4 }))
    const wet = b.derived('wetProbs', ops.where(rain, ops.where(sprinkler, 0.99, 0.8), ops.where(sprinkler, 0.9, 0.05)))
    b.observe('wet', d.bernoulli({ probs: wet }), b.data('wet', { support: 'binary' }))
  })

  it('is exact on a tree of Bernoulli nodes: it explains away', () => {
    const m = bpSpecMethod(net.bind({ wet: 1 }))
    const s = session(m, { config: { schedule: 'tree' }, steps: 20, mode: 'eager' })
    expect(s.status).toBe('converged')
    const joint = (r: number, q: number): number =>
      (r ? 0.3 : 0.7) * (q ? 0.4 : 0.6) * (r ? (q ? 0.99 : 0.8) : q ? 0.9 : 0.05)
    const z = joint(0, 0) + joint(0, 1) + joint(1, 0) + joint(1, 1)
    const post = m.posterior?.(s.state, s.config)
    expect(Math.abs((toFlat(post?.mean('rain') as never)[0] as number) - (joint(1, 0) + joint(1, 1)) / z)).toBeLessThan(
      1e-10,
    )
    expect(
      Math.abs((toFlat(post?.mean('sprinkler') as never)[0] as number) - (joint(0, 1) + joint(1, 1)) / z),
    ).toBeLessThan(1e-10)
    expect(post?.logEvidence?.method).toBe('bethe')
    expect(Math.abs((post?.logEvidence?.value ?? Number.NaN) - Math.log(z))).toBeLessThan(1e-10)
  })

  it('refuses a model with continuous latents, saying which', () => {
    const m = bpSpecMethod(normalMean.bind({ y: Y4 }))
    expect(m.check?.(analyseSpec(normalMean))).toMatchObject({
      applies: false,
      missing: ['finite discrete latents only'],
    })
    expect(m.check?.(analyseSpec(normalMean)).reason).toMatch(/'mu' is continuous/)
  })
})

describe('what a message-passing method says about a model it does not apply to', () => {
  it('a custom node is opaque, and the node and the reason are in the verdict and in the error of a run', () => {
    const m = defineModel('custom', (b) => {
      const warped = b.custom('warp', ([a]) => (a as NamedValue).value, [b.latent('mu', d.normal(0, 1))])
      b.observe('y', d.normal(warped, 1), b.data('y', { dims: ['row'] }))
    })
    const ep = epSpecMethod(m.bind({ y: [1, 2] }))
    const v = ep.check?.(analyseSpec(m))
    expect(v).toMatchObject({ applies: false, missing: ['exponential-family factors'] })
    expect(v?.reason).toMatch(/'y' is opaque to message passing/)
    expect(() => session(ep, { steps: 5, mode: 'eager' })).toThrow(/opaque/)
  })

  it('a log-density term, a param and an unbounded discrete latent are named', () => {
    const term = defineModel('term', (b) => {
      const x = b.latent('x', d.normal(0, 1))
      b.factor('penalty', ops.neg(ops.square(x)))
    })
    expect(epSpecMethod(term.bind({})).check?.(analyseSpec(term)).reason).toMatch(/'penalty'.*no message operator/)
    const counts = defineModel('counts', (b) => {
      b.observe('y', d.normal(b.latent('n', d.poisson(3)), 1), b.data('y', { dims: ['row'] }))
    })
    expect(epSpecMethod(counts.bind({ y: [1] })).check?.(analyseSpec(counts)).reason).toMatch(
      /unbounded discrete latent 'n'/,
    )
  })

  it('a random scale, a non-affine operator and a closure program say why from the compiler and the analysis', () => {
    const scale = defineModel('scale', (b) => {
      b.observe('y', d.normal(0, b.latent('s', d.gamma(2, 2))), b.data('y', { dims: ['row'] }))
    })
    const v = epSpecMethod(scale.bind({ y: [1, 2] })).check?.(analyseSpec(scale))
    expect(v?.applies).toBe(false)
    expect(v?.reason).toMatch(/y: the scale of 'y' depends on a random value/)
    const program = epSpecMethod(() => undefined)
    expect(program.check?.(analyseSpec(scale))).toMatchObject({ applies: false, missing: ['a model spec'] })
    expect(() => session(program, { steps: 5, mode: 'eager' })).toThrow(/closure program has no factor graph/)
  })
})

describe('the model registry offers message passing by name', () => {
  const models = modelRegistry({ probit: { definition: probitModel, data: { X, y } } })

  it('lists ep, vmp and bp among the kinds, and builds ep/probit with only knob values from the wire', () => {
    expect(Object.keys(methodKinds)).toEqual(expect.arrayContaining(['ep', 'vmp', 'bp']))
    const m = inferenceMethodFor(models, 'ep', 'probit')
    expect(m.name).toBe('ep/probit')
    const direct = epSpecMethod(models.need('probit'))
    const knobs = JSON.parse(JSON.stringify({ tol: 1e-10, damping: 0 }))
    const a = session(m, { config: knobs, seed: 3, steps: 300, mode: 'eager' })
    const b = session(direct, { config: knobs, seed: 3, steps: 300, mode: 'eager' })
    expect(a.t).toBe(b.t)
    expect(toFlat(m.posterior?.(a.state, a.config)?.mean('w') as never)).toEqual(
      toFlat(direct.posterior?.(b.state, b.config)?.mean('w') as never),
    )
  })
})
