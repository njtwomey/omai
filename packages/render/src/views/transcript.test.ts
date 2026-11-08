import { transcript } from 'omai-engine/exposition'
import { describe, expect, it } from 'vitest'
import { heaviestTokens, transcriptScene, transcriptTurns } from './transcript.ts'

const r = transcript({
  tokens: [
    { text: 'Is', role: 'user' },
    { text: ' it', role: 'user' },
    { text: ' so', role: 'assistant' },
  ],
  tracks: [
    { name: 'head 1', weights: [1, 4, 2] },
    { name: 'head 2', weights: [0, 0, 0] },
  ],
  emitted: 2,
})

describe('the transcript views', () => {
  it('normalises the weights of a track to levels and flags the emitted token', () => {
    const turns = transcriptTurns(r)
    expect(turns.map((t) => t.role)).toEqual(['user', 'assistant'])
    expect(turns[0]?.spans.map((s) => s.level)).toEqual([0.25, 1])
    expect(turns[1]?.spans[0]?.emitted).toBe(true)
    expect(turns[1]?.spans[0]?.title).toBe('"so", head 1 2, emitted at this step')
    // A track with no weight highlights nothing, and a missing track too.
    expect(transcriptTurns(r, 1)[0]?.spans.map((s) => s.level)).toEqual([0, 0])
    expect(transcriptTurns(r, 7)[0]?.spans[1]?.title).toBe('"it"')
  })

  it('names the heaviest tokens, heaviest first and by position on a tie', () => {
    expect(heaviestTokens(r, 0, 2)).toEqual([
      { text: 'it', index: 1, weight: 4 },
      { text: 'so', index: 2, weight: 2 },
    ])
    expect(heaviestTokens(r, 1)).toEqual([])
  })

  it('draws a still: one line per turn, wrapped, with the emitted token marked', () => {
    const s = transcriptScene(r)
    const lines = s.children.map((n) => (n.kind === 'text' ? n.text : ''))
    expect(lines).toEqual(['user: Is it', 'assistant: ▸ so'])
    expect(s.title).toBe('Transcript of 3 tokens')
  })
})
