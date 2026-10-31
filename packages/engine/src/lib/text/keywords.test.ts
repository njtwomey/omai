import { describe, expect, it } from 'vitest'
import { pagerank } from '../graph/index.ts'
import { overlapSimilarity, rake, sentenceGraph, textRank, textRankGraph, textRankSentences } from './keywords.ts'

// The abstract and the stop list of the RAKE paper (Rose, Engel, Cramer and Cowley 2010, section 1.2.1).
const ABSTRACT =
  'Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types of systems and systems of mixed types.'
const STOP = ['of', 'over', 'the', 'are', 'for', 'a', 'and', 'be', 'can', 'in', 'all', 'these', 'to', 'used', 'given']

describe('RAKE', () => {
  const scores = new Map(rake(ABSTRACT, { stopWords: STOP }).map((r) => [r.phrase, r.score]))

  it("reproduces the paper's keyword scores", () => {
    expect(scores.get('minimal generating sets')).toBeCloseTo(8.667, 3)
    expect(scores.get('linear diophantine equations')).toBeCloseTo(8.5, 12)
    expect(scores.get('minimal supporting set')).toBeCloseTo(7.667, 3)
    expect(scores.get('minimal set')).toBeCloseTo(4.667, 3)
    expect(scores.get('linear constraints')).toBeCloseTo(4.5, 12)
    expect(scores.get('natural numbers')).toBe(4)
    expect(scores.get('strict inequations')).toBe(4)
    expect(scores.get('nonstrict inequations')).toBe(4)
    expect(scores.get('upper bounds')).toBe(4)
    // 'mixed types': mixed has degree 2 and frequency 1, types has degree 5 and frequency 3.
    expect(scores.get('mixed types')).toBeCloseTo(2 + 5 / 3, 12)
  })

  it('ranks the phrases best first and respects a length limit', () => {
    const ranked = rake(ABSTRACT, { stopWords: STOP })
    expect(ranked[0]?.phrase).toBe('minimal generating sets')
    for (let i = 1; i < ranked.length; i++) expect(ranked[i]?.score).toBeLessThanOrEqual(ranked[i - 1]?.score ?? 0)
    expect(rake(ABSTRACT, { stopWords: STOP, maxWords: 2 }).every((r) => r.phrase.split(' ').length <= 2)).toBe(true)
  })
})

describe('TextRank', () => {
  const tokens = 'graph model graph node graph edge graph path'.split(' ')

  it('links words within the window and counts co-occurrences', () => {
    const g = textRankGraph(['a', 'b', 'a', 'b'], { keep: () => true, window: 2 })
    expect(g.words).toEqual(['a', 'b'])
    expect(g.graph.edges.map((e) => e.weight)).toEqual([3])
  })

  it('ranks the hub word first, with scores that sum to one', () => {
    const r = textRank(tokens, { keep: () => true, topK: 3 })
    expect(r.words[0]?.word).toBe('graph')
    const g = textRankGraph(tokens, { keep: () => true })
    expect(pagerank(g.graph).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
  })

  it('merges adjacent top words into phrases', () => {
    const text = 'machine learning models need data and machine learning needs compute'.split(' ')
    const r = textRank(text, { topK: 3, merge: true })
    expect(r.phrases).toContain('machine learning')
  })

  it('ranks the central sentence of a similarity graph highest', () => {
    const s = [
      ['cat', 'sat', 'mat', 'rug'],
      ['cat', 'sat', 'rug', 'mat', 'soft'],
      ['dog', 'ran', 'far', 'away'],
      ['cat', 'mat', 'rug', 'sat'],
    ]
    const r = textRankSentences(s, 2)
    expect(r.chosen.every((i) => i !== 2)).toBe(true)
    expect(r.scores[2]).toBeLessThan(r.scores[0] ?? 0)
    expect(sentenceGraph(s).edges.length).toBe(6 - 3)
  })

  it('has the similarity of the paper', () => {
    expect(overlapSimilarity(['a', 'b', 'c'], ['b', 'c', 'd', 'e'])).toBeCloseTo(2 / (Math.log(3) + Math.log(4)), 12)
    expect(overlapSimilarity(['a'], ['a', 'b'])).toBe(0)
  })
})
