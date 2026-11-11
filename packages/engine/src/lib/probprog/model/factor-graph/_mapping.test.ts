// The protocol of a mapping table: an entry that tries a node and answers NOT_MINE leaves the graph as it was.
import { describe, expect, it } from 'vitest'
import { OmaiError } from '../../../base/index.ts'
import { defaultOperators as factorOperators } from '../../../message-passing/index.ts'
import * as d from '../dists/index.ts'
import { defineModel } from '../index.ts'
import { defaultOperators } from '../operators.ts'
import { Context } from './_context.ts'
import { attempt, type MappingTable, mapFirst, NOT_MINE, probe } from './_mapping.ts'
import { Refusal } from './_refusal.ts'

const cxOf = (): Context => {
  const model = defineModel('probe', (m) => {
    const n = m.range('n')
    m.latent('a', d.normal(0, 1), { dims: [n] })
  })
  return new Context(model.bind({}, { sizes: { n: 2 } }), defaultOperators(), factorOperators(), {
    vectorise: true,
    packScalars: false,
    denseScales: true,
  })
}

/** What a graph holds, to compare before and after. */
const shape = (cx: Context): unknown => ({
  variables: cx.g.variables.map((v) => v.name),
  factors: cx.g.factors.map((f) => f.id),
  auxiliary: [...cx.g.auxiliary],
  atoms: [...cx.g.atoms.keys()],
  applied: [...cx.applied.keys()],
  cells: [...cx.cells.keys()],
})

/** Something an evaluation leaves behind: an auxiliary variable with a factor, an atom, an apply chain and a cached cell. */
const leaveResidue = (cx: Context, site: string): string => {
  const name = cx.g.auxName(site, undefined)
  cx.g.addVariable({ name, family: { kind: 'gaussian' } })
  cx.g.addFactor({ operator: 'dot-constant', args: [name, 'x'], params: { a: [1] } }, name)
  cx.g.varAtom(name)
  cx.applied.set(name, { input: 'x', steps: [] })
  cx.cells.set(`${site}|probe`, [])
  return name
}

describe('GraphBuilder checkpoint and restore', () => {
  it('cuts the graph back to what it held, with the names, the components and the counters of auxiliary names', () => {
    const cx = cxOf()
    cx.g.addVariable({ name: 'x', family: { kind: 'gaussian' } })
    const first = cx.g.auxName('y', undefined)
    const before = cx.g.checkpoint()
    const snapshot = shape(cx)
    expect(cx.g.changedSince(before)).toBe(false)
    const made = leaveResidue(cx, 'y')
    cx.g.component('x', 0, 2)
    expect(cx.g.changedSince(before)).toBe(true)
    cx.g.restore(before)
    const graphOf = (x: unknown): unknown => ({ ...(x as object), applied: undefined, cells: undefined })
    expect(graphOf(shape(cx))).toEqual(graphOf(snapshot))
    // the name `y#1` is free again (the variable was removed with its name) and the next auxiliary name repeats
    expect(first).toBe('y#0')
    expect(made).toBe('y#1')
    expect(cx.g.auxName('y', undefined)).toBe(made)
    expect(() => cx.g.addVariable({ name: made, family: { kind: 'gaussian' } })).not.toThrow()
    expect(() => cx.g.addVariable({ name: 'x[0]', family: { kind: 'gaussian' } })).not.toThrow()
  })
})

describe('attempt and probe', () => {
  it('a failed attempt that made things leaves the graph, the apply chains and the cached cells as they were', () => {
    const cx = cxOf()
    const before = shape(cx)
    const answer = attempt(cx, () => {
      leaveResidue(cx, 'y')
      return NOT_MINE
    })
    expect(answer).toBe(NOT_MINE)
    expect(shape(cx)).toEqual(before)
  })

  it('a successful attempt keeps what it made', () => {
    const cx = cxOf()
    const answer = attempt(cx, () => leaveResidue(cx, 'y'))
    expect(answer).toBe('y#0')
    expect((shape(cx) as { variables: string[] }).variables).toEqual(['y#0'])
  })

  it('a failed attempt that made nothing keeps the cells it cached (the next entry reads them)', () => {
    const cx = cxOf()
    const answer = attempt(cx, () => {
      cx.cells.set('y|loc', [])
      return NOT_MINE
    })
    expect(answer).toBe(NOT_MINE)
    expect(cx.cells.has('y|loc')).toBe(true)
  })

  it('a probe turns a refusal into NOT_MINE and takes out what the evaluation made before it refused', () => {
    const cx = cxOf()
    const before = shape(cx)
    const answer = probe(cx, () => {
      leaveResidue(cx, 'y')
      throw new Refusal('operator', 'not an affine form')
    })
    expect(answer).toBe(NOT_MINE)
    expect(shape(cx)).toEqual(before)
    expect(probe(cx, () => 7)).toBe(7)
  })

  it('a probe passes an error that is not a refusal', () => {
    const cx = cxOf()
    expect(() =>
      probe(cx, () => {
        throw new OmaiError('value', 'a bug')
      }),
    ).toThrow('a bug')
  })
})

describe('mapFirst', () => {
  const table = (log: string[], tries: boolean): MappingTable<[string], string> => [
    {
      name: 'tries',
      maps: 'recognises by trying: makes an auxiliary variable and then finds the node is not its pattern',
      requires: [],
      matches: () => tries,
      map: (cx) => {
        leaveResidue(cx, 'y')
        log.push('tries')
        return NOT_MINE
      },
    },
    {
      name: 'refuses-b',
      maps: 'only for b',
      requires: [],
      matches: (_cx, node) => node === 'b',
      map: () => {
        throw new Refusal('structure', 'b is refused')
      },
    },
    {
      name: 'takes-the-rest',
      maps: 'everything that is left',
      requires: [],
      map: (cx, node) => {
        log.push('takes')
        return leaveResidue(cx, node)
      },
    },
  ]

  it('takes the first entry that matches and maps, so a graph that follows a failed attempt is the graph without that attempt', () => {
    const [withTry, without] = [cxOf(), cxOf()]
    const log: string[] = []
    const a = mapFirst(table(log, true), withTry, 'x', 'a')
    const b = mapFirst(table([], false), without, 'x', 'a')
    expect(log).toEqual(['tries', 'takes'])
    expect(a).toBe(b)
    expect(shape(withTry)).toEqual(shape(without))
  })

  it('stops at the first entry that takes the node: an entry after it is not asked', () => {
    const cx = cxOf()
    expect(() => mapFirst(table([], false), cx, 'x', 'b')).toThrow('b is refused')
  })

  it('reports a table that takes nothing as a bug of the table', () => {
    expect(() => mapFirst([], cxOf(), 'x')).toThrow(/no mapping of the x/)
  })
})
