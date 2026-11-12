import { describe, expect, it } from 'vitest'
import { type AtlasCode, readAtlas } from './ep-atlas.ts'

// The atlas against a stand-in for the code: two atoms (one runs, one is refused) and one test file that holds one text.
const code: AtlasCode = {
  atoms: new Map([
    ['gaussian', true],
    ['studentish', false],
  ]),
  fileHas: (path, needle) => path === 'a/b.test.ts' && (needle === undefined || needle === 'it holds this'),
}

const header = '| parameter | factor | ran by | status | note |\n| --- | --- | --- | --- | --- |'
const table = (...rows: string[]): string => ['# Atlas', '', header, ...rows, '', 'prose'].join('\n')
const row = (ran: string, status: string): string => `| mean | \`prior\` | ${ran} | ${status} | n |`

describe('ep-atlas', () => {
  it('a row that names an atom that runs, or a test that holds its text, is `runs` and agrees with its status word', () => {
    const r = readAtlas(table(row('`atom:gaussian`', 'runs'), row('`test:a/b.test.ts#it holds this`', 'runs')), code)
    expect(r.problems).toEqual([])
    expect(r.rows.map((x) => x.status)).toEqual(['runs', 'runs'])
    expect(r.rows[0]?.atoms).toEqual(['gaussian'])
    expect(r.rows[1]?.tests).toEqual(['a/b.test.ts#it holds this'])
  })

  it('a row that says `not yet` is `planned`, and a `runs` written beside it is drift', () => {
    const ok = readAtlas(table(row('not yet', 'planned')), code)
    expect(ok.problems).toEqual([])
    expect(ok.rows[0]?.status).toBe('planned')
    const drift = readAtlas(table(row('not yet', 'runs')), code)
    expect(drift.problems).toHaveLength(1)
    expect(drift.problems[0]).toContain("status says 'runs', the code says 'planned'")
  })

  it('a stale status word is drift, and the rewritten text carries the code word', () => {
    const r = readAtlas(table(row('`atom:gaussian`', '**planned**')), code)
    expect(r.problems[0]).toContain("status says 'planned', the code says 'runs'")
    expect(r.unfixable).toEqual([])
    expect(r.text).toMatch(/\| runs\s+\|/)
    expect(readAtlas(r.text, code).problems).toEqual([])
  })

  it('an atom that does not run, an atom that does not exist, a missing file and a missing text each break the row', () => {
    const r = readAtlas(
      table(
        row('`atom:studentish`', 'broken'),
        row('`atom:nope`', 'broken'),
        row('`test:a/missing.test.ts`', 'broken'),
        row('`test:a/b.test.ts#absent`', 'broken'),
      ),
      code,
    )
    expect(r.rows.map((x) => x.status)).toEqual(['broken', 'broken', 'broken', 'broken'])
    expect(r.problems).toHaveLength(4)
    expect(r.unfixable).toEqual(r.problems)
    expect(r.problems[0]).toContain('does not run under EP')
    expect(r.problems[1]).toContain('does not exist')
    expect(r.problems[2]).toContain('does not exist')
    expect(r.problems[3]).toContain("does not hold 'absent'")
  })

  it('a row with no reference and no `not yet`, or with both, is a problem', () => {
    const r = readAtlas(table(row('', 'runs'), row('`atom:gaussian` not yet', 'runs')), code)
    expect(r.problems.some((p) => p.includes('names no atom or test'))).toBe(true)
    expect(r.problems.some((p) => p.includes('also names what runs it'))).toBe(true)
  })

  it('tables without the two columns and the prose are left as they are', () => {
    const plain = '# T\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\ntext `atom:gaussian`'
    const r = readAtlas(plain, code)
    expect(r.rows).toEqual([])
    expect(r.text).toBe(plain)
  })

  it('only the status cell changes: the other cells keep their text, the table is padded to its widest cell', () => {
    const r = readAtlas(
      table(row('`atom:gaussian`', 'planned'), '| a much longer parameter | `f` | not yet | planned | |'),
      code,
    )
    const lines = r.text.split('\n').filter((l) => l.startsWith('|'))
    expect(new Set(lines.map((l) => l.length)).size).toBe(1)
    expect(lines[2]).toContain('`atom:gaussian`')
    expect(lines[2]).toContain('| n')
  })
})
