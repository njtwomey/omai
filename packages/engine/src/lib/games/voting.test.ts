import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import {
  borda,
  condorcetLoser,
  condorcetParadoxProfile,
  condorcetWinner,
  copeland,
  instantRunoff,
  pairwiseMatrix,
  plurality,
  profile,
  smithSet,
  socialChoiceParadoxes,
  tennesseeProfile,
  withoutCandidate,
} from './index.ts'

describe('voting', () => {
  const tn = tennesseeProfile()

  it('reproduces the Tennessee capital example', () => {
    expect(plurality(tn).scores).toEqual([42, 26, 15, 17])
    expect(borda(tn).scores).toEqual([126, 194, 173, 107])
    expect(condorcetWinner(tn)).toBe(1)
    expect(condorcetLoser(tn)).toBe(0)
    expect(copeland(tn).winners).toEqual([1])
    const irv = instantRunoff(tn)
    expect(irv.winner).toBe(3)
    expect(irv.rounds.map((r) => r.eliminated)).toEqual([2, 1, null])
    expect(pairwiseMatrix(tn)[1]?.[0]).toBe(58)
  })

  it('finds the paradoxes', () => {
    const r = socialChoiceParadoxes(tn)
    expect(r.winners).toEqual({ plurality: 0, borda: 1, copeland: 1, instantRunoff: 3 })
    expect(r.condorcetLoserWins).toEqual(['plurality'])
    expect(r.majorityCycle).toBe(false)
    // Without Memphis, its voters' second choice Nashville wins the instant runoff.
    expect(r.spoilers).toContainEqual({ rule: 'instantRunoff', removed: 0, before: 3, after: 1 })
    const cycle = socialChoiceParadoxes(condorcetParadoxProfile())
    expect(cycle.condorcetWinner).toBeNull()
    expect(smithSet(condorcetParadoxProfile())).toEqual([0, 1, 2])
  })

  it('keeps indices straight when candidates are removed', () => {
    const p = withoutCandidate(tn, 2)
    expect(p.candidates).toEqual(['Memphis', 'Nashville', 'Knoxville'])
    expect(p.ballots[2]).toEqual([2, 1, 0])
    expect(() => profile(['a'], [['b']])).toThrow()
    expect(() => profile(['a', 'b'], [['a', 'a']])).toThrow()
  })

  it('the Condorcet winner, when there is one, is the Smith set and wins Copeland', () => {
    const ballot = fc.shuffledSubarray([0, 1, 2, 3], { minLength: 4, maxLength: 4 })
    fc.assert(
      fc.property(fc.array(ballot, { minLength: 1, maxLength: 9 }), (bs) => {
        const p = profile(['a', 'b', 'c', 'd'], bs)
        const cw = condorcetWinner(p)
        if (cw !== null) {
          expect(smithSet(p)).toEqual([cw])
          expect(copeland(p).winners).toEqual([cw])
        }
        let votes = 0
        for (const c of p.counts) votes += c
        const s = borda(p).scores
        let total = 0
        for (const v of s) total += v
        expect(total).toBe(votes * 6)
      }),
    )
  })
})
