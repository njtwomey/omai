// The TrueSkill page's compute: the messages of EP on the games so far as a record, the beliefs as games are added, and
// NUTS as the reference.
import { describe, expect, it } from 'vitest'
import { snapshotOf } from '../../../shared/smoke.ts'
import { beliefsAlong, MAX_GAMES, messageRun, nutsRun, tournamentOf } from './compute.ts'

describe('the TrueSkill runs', () => {
  const t = tournamentOf(2)

  it('records the messages on the factor graph of the games so far, with the players’ beliefs at each step', () => {
    const r = messageRun(t, 6)
    expect(r.steps.length).toBeGreaterThan(5)
    const ids = r.structure.nodes.map((n) => n.id)
    expect(ids).toContain('factor:win_6')
    expect(ids).not.toContain('factor:win_7')
    expect(r.steps.at(-1)?.nodes['skill_A']?.sd).toBeLessThan(1)
  })

  it('narrows the beliefs as games are added, and puts the best player on top at the end', () => {
    const along = beliefsAlong(t)
    expect(along).toHaveLength(MAX_GAMES + 1)
    expect(along[0]?.[0]?.sd).toBe(1)
    expect((along[MAX_GAMES]?.[0]?.sd ?? 1) < (along[1]?.[0]?.sd ?? 0)).toBe(true)
    expect((along[MAX_GAMES]?.[0]?.mean ?? 0) > (along[MAX_GAMES]?.[2]?.mean ?? 0)).toBe(true)
  })

  it('reads NUTS as a belief per player', () => {
    const [spec] = nutsRun(t, 6)
    const k = snapshotOf(spec as NonNullable<typeof spec>, { numWarmup: 30, numDraws: 30 }, 100)
    expect(k).toHaveLength(3)
    expect(k[0]?.sd).toBeGreaterThan(0)
  })
})
