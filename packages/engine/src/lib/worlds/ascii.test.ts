import { describe, expect, it } from 'vitest'
import { isWhiteboxError } from '../base/index.ts'
import { fromAscii, fromRows, terrainOf, toAscii } from './index.ts'

const code = (f: () => unknown): string | undefined => {
  try {
    f()
  } catch (e) {
    return isWhiteboxError(e) ? e.code : 'other'
  }
  return undefined
}

describe('ASCII maps', () => {
  it('rejects ragged rows, unknown characters and malformed wall drawings', () => {
    expect(code(() => fromAscii(['..', '.']))).toBe('value')
    expect(code(() => fromAscii([]))).toBe('value')
    expect(code(() => fromAscii(['.?']))).toBe('value')
    expect(code(() => fromAscii(['####', '#..#', '####'], { format: 'walls' }))).toBe('value')
    expect(code(() => fromAscii(['###', '#.#', '###'], { format: 'walls', topology: 'hex' }))).toBe('value')
    expect(code(() => toAscii(fromAscii(['..'], { topology: 'hex' }), { format: 'walls' }))).toBe('value')
    expect(code(() => fromRows([[0, 99]]))).toBe('value')
    expect(code(() => fromRows([[0], [0, 0]]))).toBe('value')
  })

  it('reads custom legends, terrains and landmarks, and writes them back', () => {
    const legend = {
      F: { terrain: 'ice' },
      R: { terrain: 'ice', landmark: 'R' },
      B: { terrain: 'ice', landmark: 'B' },
      k: { terrain: 'ice', item: { kind: 'key' as const, colour: 'red' } },
      d: { terrain: 'ice', item: { kind: 'door' as const, colour: 'red' } },
    }
    const terrains = [{ name: 'ice', char: 'F', cost: 2 }]
    const rows = ['RFk', 'dFB']
    const w = fromAscii(rows, { legend, terrains, name: 'custom' })
    expect(w.name).toBe('custom')
    expect(w.landmarks).toEqual({ R: [0], B: [5] })
    expect(w.items).toEqual([
      { kind: 'key', cell: 2, colour: 'red', reward: 0 },
      { kind: 'door', cell: 3, colour: 'red', reward: 0 },
    ])
    expect(terrainOf(w, 1).cost).toBe(2)
    expect(toAscii(w, { legend })).toEqual(rows)
    // Without the legend, cells fall back to their terrain's character.
    expect(toAscii(w)).toEqual(['FFF', 'FFF'])
  })

  it('reads hexagonal rows with the same characters', () => {
    const w = fromAscii(['S.', '.G'], { topology: 'hex' })
    expect([w.topology.kind, w.starts, w.goals]).toEqual(['hex', [0], [3]])
    expect(fromRows([[0, 1]], { topology: 'hex', name: 'h' }).topology.kind).toBe('hex')
  })
})
