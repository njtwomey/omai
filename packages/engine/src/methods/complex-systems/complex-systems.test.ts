import { describe, expect, it } from 'vitest'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { schelling } from './agent-based.ts'
import { elementaryAutomaton, gameOfLife } from './cellular-automata.ts'
import { flocking } from './flocking.ts'

describe('complex systems', () => {
  it('draws rule 90 as Sierpinski’s triangle and keeps a glider at five cells', () => {
    const r = elementaryAutomaton(90, { width: 15, steps: 3 })
    expect(r.rows.map((row) => row.join(''))).toEqual([
      '000000010000000',
      '000000101000000',
      '000001000100000',
      '000010101010000',
    ])
    expect(recordsOf(r.exposition, 'boardSteps')[0]?.steps).toHaveLength(4)
    const life = gameOfLife({ rows: 12, cols: 12, pattern: 'glider', generations: 8 })
    expect(life.population.every((p) => p === 5)).toBe(true)
    expect(gameOfLife({ generations: 30 }).population.at(-1)).toBeGreaterThan(36)
  })

  it('segregates a town with mild preferences', () => {
    const r = schelling({ key: key(1), size: 20, threshold: 0.5 })
    expect(r.similarity.at(-1)).toBeGreaterThan((r.similarity[0] ?? 1) + 0.1)
    expect(r.unhappy.at(-1)).toBeLessThan(r.unhappy[0] ?? 0)
  })
})

describe('flocking', () => {
  it('records positions, headings and polarisation per frame, the flock aligning', () => {
    const r = flocking({ count: 60, steps: 200, every: 4 })
    const [rec] = r.exposition
    if (rec?.kind !== 'agentSteps') throw new Error('expected agentSteps')
    expect(rec.frames).toHaveLength(51)
    expect(rec.frames[0]?.heading).toHaveLength(60)
    expect(r.polarisation.at(-1)).toBeGreaterThan(r.polarisation[0] ?? 1)
  })
})
