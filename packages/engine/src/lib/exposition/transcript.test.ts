import { describe, expect, it } from 'vitest'
import { transcript, turnsOf } from './transcript.ts'

const tokens = [
  { text: 'a', role: 'user' },
  { text: ' b', role: 'user' },
  { text: ' c', role: 'assistant' },
  { text: ' d' },
]
describe('the transcript record', () => {
  it('groups consecutive tokens of one role into turns', () => {
    expect(turnsOf(transcript({ tokens }))).toEqual([
      { role: 'user', from: 0, to: 2 },
      { role: 'assistant', from: 2, to: 3 },
      { role: undefined, from: 3, to: 4 },
    ])
  })

  it('checks the tracks and the emitted index against the tokens', () => {
    expect(() => transcript({ tokens, tracks: [{ name: 'h', weights: [1, 2] }] })).toThrow(/tracks\[0\] has 2/)
    expect(() => transcript({ tokens, tracks: [{ name: 'h', weights: [1, 2, -1, 0] }] })).toThrow(/not a weight/)
    expect(() => transcript({ tokens, emitted: 4 })).toThrow(/not one of the 4/)
    expect(() => transcript({ tokens, emitted: 1.5 })).toThrow(/emitted/)
    expect(transcript({ tokens, emitted: 3 }).emitted).toBe(3)
  })
})
