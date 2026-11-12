// The graph corpus: every atom and benchmark model of the library compiled (compile only, no run) as EP compiles it and with the rows
// expanded, and the shape of each graph written down: the number of variables and factors, the issues, the operators with their
// counts. A change to a mapping that changes a graph, by an operator more or fewer or an issue, shows here as a line that moved; the
// exact graphs are the digest's (`make ep-digest-check`), this is the cheap guard that runs with the tests.
import { describe, expect, it } from 'vitest'
import { atomBound, epAtoms } from '../../ep/index.ts'
import { benchmarkModels } from '../../models/index.ts'
import { analyseSpec } from '../analysis.ts'
import type { BoundModel } from '../bind.ts'
import { toFactorGraph } from '../factors.ts'

/** The items of the corpus, by key. */
function* corpus(): Generator<{ readonly key: string; readonly bound: BoundModel }> {
  for (const a of epAtoms) yield { key: `atom/${a.name}`, bound: atomBound(a.problem(a.defaults)) }
  for (const m of benchmarkModels()) if (m.definition !== undefined) yield { key: `model/${m.name}`, bound: m.bound() }
}

/** One line for a compiled graph: `key plates: V variables, F factors, I issues; operator xN ...`. */
function line(key: string, bound: BoundModel, plates: 'vectorise' | 'expand'): string {
  const c = toFactorGraph(bound, { analysis: analyseSpec(bound), plates, packScalars: true })
  const counts = new Map<string, number>()
  for (const f of c.graph?.factors ?? []) counts.set(f.operator, (counts.get(f.operator) ?? 0) + 1)
  const operators = [...counts].sort(([a], [b]) => (a < b ? -1 : 1)).map(([op, n]) => `${op} x${n}`)
  const shape = `${c.graph?.variables.length ?? '-'} variables, ${c.graph?.factors.length ?? '-'} factors`
  const issues = c.issues.map((i) => `${i.code}`).join(',')
  return `${key} ${plates}: ${shape}, ${c.issues.length} issues${issues === '' ? '' : ` (${issues})`}; ${operators.join(' ')}`
}

describe('the graph corpus', () => {
  it('compiles every atom and benchmark model to the graph written down (variables, factors, issues, operators)', async () => {
    const lines: string[] = []
    for (const { key, bound } of corpus())
      for (const plates of ['vectorise', 'expand'] as const) lines.push(line(key, bound, plates))
    await expect(`${lines.join('\n')}\n`).toMatchFileSnapshot('./graph-corpus.snap.txt')
  })

  it('has every atom and every benchmark model with a definition', () => {
    const keys = [...corpus()].map((item) => item.key)
    expect(new Set(keys).size).toBe(keys.length)
    expect(keys.filter((k) => k.startsWith('atom/')).length).toBe(epAtoms.length)
  })
})
