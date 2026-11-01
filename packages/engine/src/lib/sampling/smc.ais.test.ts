import { describe, expect, it } from 'vitest'
import { normal } from '../dists/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { policies, recorder, runAlgorithm, stateToJson } from '../trace/index.ts'
import { annealingLadders } from './ais-options.ts'
import { aisAlgorithm, aisContract, annealedImportanceSampling } from './smc.ts'
import { proposalFromDistribution } from './target.ts'

const prior = proposalFromDistribution(normal(0, 3))
const ll = (x: Float64Array): number => -0.5 * (1 - (x[0] ?? 0)) ** 2 - 0.5 * Math.log(2 * Math.PI)
const exactLogZ = -0.5 * Math.log(2 * Math.PI * 10) - 0.05

describe('annealedImportanceSampling is a run of aisAlgorithm', () => {
  // Taken from the previous implementation.
  it('gives the numbers it gave before, exactly', () => {
    const a = annealedImportanceSampling(prior, ll, {
      key: key(5),
      numParticles: 40,
      temperatures: [0, 0.1, 0.5, 1],
      mcmcSteps: 3,
    })
    expect([a.logEvidence.value, a.logEvidence.standardError, a.ess]).toEqual([
      expect.closeTo(-2.1738988017933885, 6),
      expect.closeTo(0.1472899845190776, 6),
      expect.closeTo(21.667542264154054, 6),
    ])
    expect(toFlat(a.particles).slice(0, 2)).toEqual([
      expect.closeTo(-0.6131504202369733, 6),
      expect.closeTo(0.7520315140146523, 6),
    ])
    expect(toFlat(a.logWeights).slice(0, 2)).toEqual([
      expect.closeTo(-2.090171497268085, 6),
      expect.closeTo(-3.011076287194096, 6),
    ])
    const b = annealedImportanceSampling(prior, ll, { key: key(6), numParticles: 30 })
    expect([b.logEvidence.value, b.logEvidence.standardError, b.ess]).toEqual([
      expect.closeTo(-2.1271740720577217, 6),
      expect.closeTo(0.07180034876735167, 6),
      expect.closeTo(26.09822613496709, 6),
    ])
  })
})

describe('aisAlgorithm', () => {
  const make = () => aisAlgorithm(prior, ll, { key: key(5), numParticles: 40, temperatures: [0, 0.1, 0.5, 1] })

  it('has a state that is JSON, and a run resumes to the same end', () => {
    const alg = make()
    const whole = runAlgorithm(alg, { maxSteps: 10 })
    const half = runAlgorithm(alg, { maxSteps: 1 }).state
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(half)))
    const rest = runAlgorithm(alg, { maxSteps: 9, start: { state: stored, step: 1, serialised: true } })
    expect(stateToJson(rest.state)).toEqual(stateToJson(whole.state))
    expect(whole.state.rung).toBe(3)
    expect(whole.status).toBe('converged')
  })

  it('logs the temperature, ESS, mean log weight and evidence at every rung', () => {
    const rec = recorder()
    runAlgorithm(make(), {
      maxSteps: 10,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['state'], detail: 'full' },
    })
    const metric = (n: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === n ? [e.value] : []))
    expect(metric('ais/beta')).toEqual([0.1, 0.5, 1])
    expect(metric('ais/ess')).toHaveLength(3)
    expect(metric('ais/log-z')).toHaveLength(3)
    expect(Object.keys(aisContract)).toContain('ais/particles')
  })

  it('the effective sample size never exceeds the particle count, and the evidence is near the exact one', () => {
    const rec = recorder()
    const s = runAlgorithm(
      aisAlgorithm(prior, ll, {
        key: key(1),
        numParticles: 400,
        temperatures: annealingLadders.make({ kind: 'power', rungs: 30 }) as number[],
      }),
      { maxSteps: 100, tracer: rec, policy: { ...policies.minimal, metrics: true, detail: 'full' } },
    )
    const ess = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'ais/ess' ? [e.value] : []))
    expect(ess.every((v) => v >= 1 && v <= 400 + 1e-9)).toBe(true)
    const logZ = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'ais/log-z' ? [e.value] : []))
    expect(Math.abs((logZ.at(-1) as number) - exactLogZ)).toBeLessThan(0.15)
    expect(s.state.logWeights).toHaveLength(400)
  })

  it('rejects a bad ladder', () => {
    expect(() => aisAlgorithm(prior, ll, { key: key(0), temperatures: [0, 0.5] })).toThrow()
  })
})

describe('annealingLadders: the slot', () => {
  it('each option makes an increasing ladder from 0 to 1 that the algorithm accepts', () => {
    for (const kind of annealingLadders.names) {
      const ladder = annealingLadders.make({ kind, rungs: 8 }) as number[]
      expect(ladder[0]).toBe(0)
      expect(ladder.at(-1)).toBe(1)
      expect(ladder.every((v, i) => i === 0 || v > (ladder[i - 1] as number))).toBe(true)
      const s = runAlgorithm(aisAlgorithm(prior, ll, { key: key(2), numParticles: 10, temperatures: ladder }), {
        maxSteps: 20,
      }).state
      expect(s.rung).toBe(ladder.length - 1)
    }
    expect(annealingLadders.make({ kind: 'linear', rungs: 4 })).toEqual([0, 0.25, 0.5, 0.75, 1])
    expect(annealingLadders.where((p) => p.denseNearPrior)).toEqual(['power', 'log-spaced'])
  })
})
