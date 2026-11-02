import { describe, expect, it } from 'vitest'
import { INDEX_WORDS_MAX, INDEX_WORDS_WARN, parseIndexTag, readIndexTags } from './index-tags.ts'

const words = (n: number): string => Array.from({ length: n }, (_, i) => `w${i}`).join(' ')

describe('parseIndexTag', () => {
  it('reads the kind and the description', () => {
    expect(parseIndexTag('function Solve a linear system.')).toEqual({
      tag: { kind: 'function', description: 'Solve a linear system.' },
    })
    expect(parseIndexTag('  estimator   Two   spaces. ')).toEqual({
      tag: { kind: 'estimator', description: 'Two spaces.' },
    })
  })

  it('accepts every kind of the vocabulary', () => {
    const kinds = 'function estimator algorithm structure table module dataset distribution kernel agent'
    for (const kind of kinds.split(' ')) expect(parseIndexTag(`${kind} Something.`)).toHaveProperty('tag.kind', kind)
  })

  it('rejects an unknown kind and names the vocabulary', () => {
    const r = parseIndexTag('metric Mean squared error.')
    expect(r).toHaveProperty('error')
    expect((r as { error: string }).error).toContain("unknown @index kind 'metric'")
    expect((r as { error: string }).error).toContain('estimator')
  })

  it('rejects a missing kind and an empty description', () => {
    expect(parseIndexTag('')).toHaveProperty('error')
    expect((parseIndexTag('function') as { error: string }).error).toContain('needs a description')
    expect((parseIndexTag('table   ') as { error: string }).error).toContain('needs a description')
  })

  it('rejects a description over the maximum and accepts one at it', () => {
    expect(parseIndexTag(`function ${words(INDEX_WORDS_MAX)}`)).toHaveProperty('tag')
    const r = parseIndexTag(`function ${words(INDEX_WORDS_MAX + 1)}`)
    expect((r as { error: string }).error).toContain(`${INDEX_WORDS_MAX + 1} words`)
  })
})

describe('readIndexTags', () => {
  it('returns nothing for a comment without a tag', () => {
    expect(readIndexTags([], 'a.ts', 3)).toEqual({ problems: [] })
  })

  it('returns the tag and no problem for a valid one', () => {
    const r = readIndexTags([{ text: 'kernel Matern kernel.', line: 4 }], 'a.ts', 10)
    expect(r.index).toEqual({ kind: 'kernel', description: 'Matern kernel.' })
    expect(r.problems).toEqual([])
  })

  it('reports file lines: the comment line plus the line within the comment', () => {
    const r = readIndexTags([{ text: 'verb Do it.', line: 4 }], 'src/a.ts', 10)
    expect(r.index).toBeUndefined()
    expect(r.problems).toEqual([
      { severity: 'error', file: 'src/a.ts', line: 14, message: expect.stringContaining("unknown @index kind 'verb'") },
    ])
  })

  it('warns about a long description but keeps the tag', () => {
    const r = readIndexTags([{ text: `function ${words(INDEX_WORDS_WARN + 1)}`, line: 2 }], 'a.ts', 1)
    expect(r.index?.kind).toBe('function')
    expect(r.problems).toHaveLength(1)
    expect(r.problems[0]).toMatchObject({ severity: 'warning', line: 3 })
  })

  it('does not warn at the warning threshold', () => {
    const r = readIndexTags([{ text: `function ${words(INDEX_WORDS_WARN)}`, line: 2 }], 'a.ts', 1)
    expect(r.problems).toEqual([])
  })

  it('errors on a description over the maximum and drops the tag', () => {
    const r = readIndexTags([{ text: `function ${words(INDEX_WORDS_MAX + 1)}`, line: 2 }], 'a.ts', 1)
    expect(r.index).toBeUndefined()
    expect(r.problems[0]).toMatchObject({ severity: 'error', line: 3 })
  })

  it('reports a duplicate tag at its own line and keeps the first', () => {
    const r = readIndexTags(
      [
        { text: 'function One.', line: 2 },
        { text: 'function Two.', line: 5 },
      ],
      'a.ts',
      20,
    )
    expect(r.index?.description).toBe('One.')
    expect(r.problems).toEqual([
      { severity: 'error', file: 'a.ts', line: 25, message: expect.stringContaining('duplicate @index') },
    ])
  })
})
