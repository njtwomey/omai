import { describe, expect, it } from 'vitest'
import { componentsRecord } from 'whitebox-engine/exposition'
import { topicWordsView, topWords } from './topics.ts'

describe('topic views', () => {
  const r = componentsRecord({
    method: 'lda',
    features: ['ball', 'goal', 'vote', 'law'],
    components: [
      [0.5, 0.4, 0.05, 0.05],
      [0.05, 0.05, 0.5, 0.4],
      [0.3, 0.1, 0.3, 0.3],
    ],
  })

  it('list each topic’s most probable words, best first', () => {
    expect(topWords(r, 2).map((t) => t.map((w) => w.word))).toEqual([
      ['ball', 'goal'],
      ['vote', 'law'],
      ['ball', 'vote'],
    ])
  })

  it('draw a bar panel per topic', () => {
    const opt = topicWordsView(r, { top: 3, names: ['sport', 'politics', 'mixed'] }) as {
      grid: unknown[]
      title: { text?: string }[]
    }
    expect(opt.grid).toHaveLength(3)
    expect(JSON.stringify(opt.title)).toContain('politics')
  })
})
