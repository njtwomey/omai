import { describe, expect, it } from 'vitest'
import {
  type AllowedCycle,
  EXCEPTIONS,
  FILE_CYCLE_ALLOWLIST,
  fileCycleReport,
  type ImportEdge,
  staleExceptions,
  tierViolations,
} from './layers-rules.ts'

const edge = (from: string, to: string, typeOnly = false): ImportEdge => ({
  from,
  to,
  typeOnly,
  where: `${from} imports '${to}'`,
})

describe('layers rules', () => {
  it('fails on imports against the tiers and passes on the right ones', () => {
    const cases: readonly [ImportEdge, boolean][] = [
      [edge('context/context.ts', 'trace/index.ts'), true], // context sits above trace
      [edge('autodiff/graph.ts', 'scene/index.ts'), true], // autodiff takes scene's graph description
      [edge('losses/large-output.ts', 'information/index.ts'), true], // information is machinery
      [edge('optim/glm.ts', 'dists/index.ts'), true],
      [edge('learn/bases.ts', 'metrics/index.ts'), true], // constructed domains import anything below them
      [edge('tensor/ops/math.ts', 'random/index.ts'), false], // the foundation does not climb
      [edge('trace/session.ts', 'context/index.ts'), false],
      [edge('scene/build.ts', 'tensor/index.ts'), false],
      [edge('numerics/linalg/lu.ts', 'learn/index.ts'), false], // machinery never imports a constructed domain
      [edge('stats/tests/independence.ts', 'graph/index.ts'), false],
      [edge('metrics/audio.ts', 'signal/index.ts'), true], // a listed exception
      [edge('metrics/other.ts', 'signal/index.ts'), false], // the exception is for one file
    ]
    for (const [e, passes] of cases) expect([e.where, tierViolations([e]).length === 0]).toEqual([e.where, passes])
  })

  it('reports a listed exception nothing uses any more', () => {
    expect(staleExceptions([])).toHaveLength(EXCEPTIONS.length)
    expect(staleExceptions(EXCEPTIONS.map((e) => edge(e.from, `${e.to}/index.ts`)))).toEqual([])
  })

  it('finds file-level cycles, ignores type-only imports, and compares with the allowlist', () => {
    const loop = [edge('x/a.ts', 'x/b.ts'), edge('x/b.ts', 'x/c.ts'), edge('x/c.ts', 'x/a.ts')]
    const found = fileCycleReport(loop, [])
    expect(found.found.map((c) => [c.members, c.allowed])).toEqual([[['x/a.ts', 'x/b.ts', 'x/c.ts'], null]])
    expect(found.found[0]?.path).toEqual(['x/a.ts', 'x/b.ts', 'x/c.ts', 'x/a.ts'])
    expect(fileCycleReport([...loop.slice(0, 2), edge('x/c.ts', 'x/a.ts', true)], []).found).toEqual([])
    expect(fileCycleReport([...loop, edge('x/c.ts', 'y/a.ts'), edge('y/a.ts', 'x/c.ts')], []).found).toHaveLength(1)
    const allowed: AllowedCycle = { files: ['x/c.ts', 'x/a.ts', 'x/b.ts'], reason: 'fixture' }
    const report = fileCycleReport(loop, [allowed])
    expect(report.found[0]?.allowed).toBe(allowed)
    expect(report.stale).toEqual([])
    expect(fileCycleReport([], [allowed]).stale).toHaveLength(1)
    expect(FILE_CYCLE_ALLOWLIST.every((a) => a.files.length > 1)).toBe(true)
  })
})
