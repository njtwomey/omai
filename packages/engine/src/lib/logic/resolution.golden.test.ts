import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { consult, renameTerm, type SolveResult, solve, type Term, termToString, termVariables } from './index.ts'

interface Expected {
  readonly answers: readonly (readonly [string, string][])[]
  readonly status: string
  readonly error: string | null
  readonly output: string
}

/** Answers as [name, canonical text] pairs, fresh variables numbered `_V0`, `_V1`, ... by first occurrence. */
function canonicalAnswers(r: SolveResult): [string, string][][] {
  return r.answers.map((a) => {
    const numbering = new Map<string, string>()
    return Object.entries(a.bindings).map(([name, value]): [string, string] => {
      for (const v of termVariables(value)) if (!numbering.has(v.name)) numbering.set(v.name, `_V${numbering.size}`)
      const t = renameTerm(value, (n) => numbering.get(n) ?? n)
      return [name, termToString(t, { ignoreOps: true })]
    })
  })
}

function formal(t: Term | null): string | null {
  if (t === null) return null
  const f = t.kind === 'compound' && t.functor === 'error' && t.args.length === 2 ? (t.args[0] as Term) : t
  return termToString(f, { ignoreOps: true })
}

describe('SLD resolution against the reference Prolog interpreter', () => {
  for (const c of loadFixture('logic/resolution').cases) {
    it(c.name, () => {
      const p = c.params
      const expected = p['result'] as Expected
      const r = solve(consult(p['program'] as string), p['query'] as string, {
        maxSolutions: 50,
        maxDepth: (p['depth'] as number | undefined) ?? 10_000,
      })
      expect(r.status).toBe(expected.status)
      expect(canonicalAnswers(r)).toEqual(expected.answers)
      expect(formal(r.error)).toBe(expected.error)
      expect(r.output).toBe(expected.output)
    })
  }
})
