import { describe, expect, it } from 'vitest'
import { normal } from '../dists/index.ts'
import { key, uniformStream } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { policies, recorder, runAlgorithm, stateToJson } from '../trace/index.ts'
import {
  adaptiveRejectionAlgorithm,
  adaptiveRejectionSample,
  ratioOfUniforms,
  ratioOfUniformsAlgorithm,
  rejectionAlgorithm,
  rejectionContract,
  rejectionSample,
} from './rejection.ts'
import { rejectionSampler, rejectionSamplers } from './rejection-options.ts'
import { numericTarget, proposalFromDistribution } from './target.ts'

const gaussian = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
const q = proposalFromDistribution(normal(0, 2))
const logM = Math.log(2 * Math.sqrt(2 * Math.PI))
const density = (x: number): number => Math.exp(-0.5 * x * x)

describe('the rejection samplers give the numbers they gave before', () => {
  it('rejection sampling', () => {
    const r = rejectionSample(gaussian, q, { key: key(3), n: 40, logEnvelope: logM })
    expect([r.proposals, r.acceptanceRate, r.complete, r.envelopeViolations]).toEqual([
      86,
      expect.closeTo(0.46511627906976744, 6),
      true,
      0,
    ])
    expect(toFlat(r.samples).slice(0, 3)).toEqual([
      expect.closeTo(-0.5634241526437909, 6),
      expect.closeTo(-0.062454163929479604, 6),
      expect.closeTo(0.8551428476199869, 6),
    ])
    expect(toFlat(r.samples).reduce((a, b) => a + b, 0)).toBeCloseTo(6.711243317082557, 6)
    const capped = rejectionSample(gaussian, q, { key: key(3), n: 40, logEnvelope: logM, maxProposals: 25 })
    expect([capped.proposals, toFlat(capped.samples).length, capped.complete]).toEqual([25, 11, false])
    expect(toFlat(capped.samples).reduce((a, b) => a + b, 0)).toBeCloseTo(1.9085858447669493, 6)
  })

  it('adaptive rejection sampling', () => {
    const r = adaptiveRejectionSample(gaussian, { key: key(3), n: 40 })
    expect([r.proposals, r.acceptanceRate, r.complete]).toEqual([41, expect.closeTo(0.975609756097561, 6), true])
    expect(toFlat(r.samples).slice(0, 3)).toEqual([
      expect.closeTo(-0.3071456813742808, 6),
      expect.closeTo(-0.7003080918315663, 6),
      expect.closeTo(-1.4691422507654783, 6),
    ])
    expect(toFlat(r.samples).reduce((a, b) => a + b, 0)).toBeCloseTo(-3.239613620985094, 6)
    expect(r.abscissae).toHaveLength(10)
    expect(r.abscissae[0]).toBeCloseTo(-2.08029767972179, 6)
  })

  it('the ratio of uniforms', () => {
    const r = ratioOfUniforms(density, { key: key(3), n: 40 })
    expect([r.proposals, r.acceptanceRate, r.complete]).toEqual([55, expect.closeTo(0.7272727272727273, 6), true])
    expect(toFlat(r.samples).slice(0, 3)).toEqual([
      expect.closeTo(0.7212521398792514, 6),
      expect.closeTo(-0.4459741805584576, 6),
      expect.closeTo(0.1268123667843284, 6),
    ])
    expect(toFlat(r.samples).reduce((a, b) => a + b, 0)).toBeCloseTo(1.8771196497813716, 6)
    expect(r.bounds.vMax).toBeCloseTo(0.8663408856828039, 6)
  })
})

const algorithms = {
  rejection: () => rejectionAlgorithm(gaussian, q, { key: key(3), n: 40, logEnvelope: logM }),
  adaptive: () => adaptiveRejectionAlgorithm(gaussian, { key: key(3), n: 40 }),
  'ratio-of-uniforms': () => ratioOfUniformsAlgorithm(density, { key: key(3), n: 40 }),
} as const

describe('the rejection algorithms: state, paths, invariants', () => {
  for (const [name, make] of Object.entries(algorithms)) {
    it(`${name}: a state that is JSON resumes the same stream to the same end`, () => {
      const alg = make() as ReturnType<typeof algorithms.rejection>
      const whole = runAlgorithm(alg, { maxSteps: 1000 })
      const half = runAlgorithm(alg, { maxSteps: 17 }).state
      const stored = JSON.parse(JSON.stringify(alg.serialise?.(half)))
      const rest = runAlgorithm(alg, { maxSteps: 983, start: { state: stored, step: 17, serialised: true } })
      expect(stateToJson(rest.state)).toEqual(stateToJson(whole.state))
      expect(whole.status).toBe('converged')
    })

    it(`${name}: logs the proposal, whether it was kept, the running count and rate`, () => {
      const rec = recorder()
      runAlgorithm(make() as ReturnType<typeof algorithms.rejection>, {
        maxSteps: 6,
        tracer: rec,
        policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output', 'state'], detail: 'full' },
      })
      const metric = (n: string): number[] =>
        rec.events.flatMap((e) => (e.type === 'metric' && e.name === n ? [e.value] : []))
      expect(metric('reject/accepted')).toHaveLength(6)
      expect(metric('reject/count')).toHaveLength(6)
      expect(metric('reject/rate').every((r) => r >= 0 && r <= 1)).toBe(true)
      const counts = metric('reject/count')
      const hits = metric('reject/accepted')
      expect(counts.at(-1)).toBe(hits.reduce((a, b) => a + b, 0))
      expect(rec.events.filter((e) => e.type === 'tensor' && e.name === 'reject/proposal')).toHaveLength(6)
    })
  }

  it('the stream resumes where it stopped: uniformStream from an offset is the tail', () => {
    const all = uniformStream(key(9), 64)
    const draws = Array.from({ length: 200 }, () => all())
    for (const from of [0, 1, 63, 64, 65, 130]) expect(uniformStream(key(9), 64, from)()).toBe(draws[from])
  })

  it('adaptive: the hull only grows, and the squeeze accepts more as it does', () => {
    const alg = adaptiveRejectionAlgorithm(gaussian, { key: key(1), n: 400 })
    const s = runAlgorithm(alg, { maxSteps: 2000 }).state
    expect(s.abscissae.length).toBeGreaterThan(2)
    expect([...s.abscissae]).toEqual([...s.abscissae].sort((a, b) => a - b))
    expect(s.evaluations).toBeLessThan(s.proposals + s.abscissae.length + 3)
    expect(s.squeezed).toBeGreaterThan(0)
    expect(s.slopes.every((d, i) => i === 0 || d <= (s.slopes[i - 1] ?? 0) + 1e-9)).toBe(true)
  })

  it('rejection: the accepted draws are exactly those the proposals gave, and the budget stops it', () => {
    const alg = rejectionAlgorithm(gaussian, q, { key: key(2), n: 100, logEnvelope: 40, maxProposals: 50 })
    const r = runAlgorithm(alg, { maxSteps: 1000 })
    expect(r.status).toBe('stopped')
    expect(r.state.proposals).toBe(50)
    expect(r.state.samples.length).toBe(r.state.accepted)
  })

  it('has a contract with the paths a page reads', () => {
    expect(Object.keys(rejectionContract)).toEqual(
      expect.arrayContaining(['reject/proposal', 'reject/accepted', 'reject/rate', 'reject/envelope']),
    )
  })
})

describe('rejectionSamplers: the slot', () => {
  const problem = { key: key(5), n: 30, target: gaussian, proposal: q, density }
  it('each option runs and draws what it was asked for', () => {
    for (const kind of rejectionSamplers.names) {
      const config = kind === 'rejection' ? { kind, logEnvelope: logM } : { kind }
      const alg = rejectionSampler(config, problem)
      const s = runAlgorithm(alg, { maxSteps: 10_000 }).state
      expect(s.accepted).toBe(30)
    }
  })

  it('says what a method needs, and what it is', () => {
    expect(() => rejectionSampler({ kind: 'rejection' }, { key: key(0), n: 1 })).toThrow(/target/)
    expect(() => rejectionSampler({ kind: 'ratio-of-uniforms' }, { key: key(0), n: 1 })).toThrow(/density/)
    expect(rejectionSamplers.where((p) => p.logConcave)).toEqual(['adaptive'])
    expect(rejectionSamplers.where((p) => p.multivariate)).toEqual(['rejection'])
  })
})
