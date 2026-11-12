import { describe, expect, it } from 'vitest'
import { arrayProblems, CATEGORIES, guarded, mayShow, regionLines } from './banned-arrays.ts'

const EP = 'packages/engine/src/lib/probprog/ep/_atoms/example.ts'
const MP = 'packages/engine/src/lib/message-passing/numerics/example.ts'
const OUTSIDE = 'packages/engine/src/lib/tensor/example.ts'

const problems = (path: string, ...lines: string[]): string[] =>
  arrayProblems(path, lines.join('\n')).map((p) => `${p.line}: ${p.message.split(':')[0]}`)

describe('hand-built arrays: what is flagged', () => {
  it.each([
    ['new Array<number>(n).fill(0)', 'const a = new Array<number>(n).fill(0)'],
    ['new Array(n)', 'const a = new Array(n)'],
    ['.fill(', 'row.fill(1)'],
    ['Array.from({ length', 'const a = Array.from({ length: n }, (_, i) => i)'],
    ['Array.from({ length without a space', 'const a = Array.from({length: n}, () => 0)'],
    ['new Float64Array(', 'const buffer = new Float64Array(n)'],
    ['new Int32Array(', 'const index = new Int32Array(n)'],
    ['new Uint8Array(', 'const flags = new Uint8Array(n)'],
  ])('flags %s in the EP code paths', (_, code) => {
    expect(problems(EP, code).length).toBeGreaterThan(0)
    expect(problems(MP, code).length).toBeGreaterThan(0)
  })

  it('passes tensor code and what only looks similar', () => {
    expect(
      problems(
        EP,
        'const a = full([n], 0)',
        'const b = toFlat(arange(n))',
        'const c = Array.from(set)',
        "const d = 'new Array(n).fill(0)'",
        '// new Float64Array(n) in a comment',
        'const e = tensor([1, 2, 3])',
        'const f = [...rows]',
      ),
    ).toEqual([])
  })

  it('looks only at the guarded paths, and skips tests', () => {
    expect(problems(OUTSIDE, 'const a = new Float64Array(n)')).toEqual([])
    expect(problems('packages/engine/src/lib/probprog/ep/example.test.ts', 'const a = new Array(n).fill(0)')).toEqual(
      [],
    )
    expect(
      problems('packages/engine/src/lib/message-passing/example.property.test.ts', 'const a = new Float64Array(3)'),
    ).toEqual([])
    expect(guarded('apps/pocs/src/examples/radon-method.ts')).toBe(true)
    expect(guarded('apps/pocs/src/pocs/expectation-propagation/basic/x.ts')).toBe(true)
    expect(guarded('apps/pocs/src/shared/ep/atomSource.ts')).toBe(true)
    expect(guarded('apps/pocs/src/shared/Page.tsx')).toBe(false)
    expect(guarded('apps/site/src/gallery/recipes/x.tsx')).toBe(false)
  })
})

describe('hand-built arrays: the marker', () => {
  it('allows a line under a marker with a category and a reason', () => {
    expect(
      problems(EP, '// banned-ok: kernel: the working buffer of a sweep', 'const buffer = new Float64Array(n)'),
    ).toEqual([])
  })

  it('covers a run of flagged lines, and ends at the first line that is not flagged', () => {
    expect(
      problems(
        MP,
        '// banned-ok: kernel: buffers of one sweep',
        'const a = new Float64Array(n)',
        'const b = new Float64Array(n)',
        'let i = 0',
        'const c = new Float64Array(n)',
      ),
    ).toEqual(['5: a typed array as a value container'])
  })

  it('wants a category from the list and a reason in words', () => {
    expect(problems(EP, '// banned-ok: because', 'const a = new Array(n)')).toEqual([
      expect.stringContaining('banned-ok needs a category'),
    ])
    expect(problems(EP, '// banned-ok: kernel:', 'const a = new Array(n)')).toEqual([
      '1: banned-ok needs a reason after the category',
    ])
    for (const category of Object.keys(CATEGORIES))
      expect(problems(EP, `// banned-ok: ${category}: a reason in words`, 'const a = new Array(n)')).toEqual([])
  })

  it('reports a marker that covers nothing, and one separated from its line', () => {
    expect(problems(EP, '// banned-ok: kernel: a buffer', 'const a = full([n], 0)')).toEqual([
      '1: banned-ok marker that no flagged line follows',
    ])
    expect(problems(EP, '// banned-ok: kernel: a buffer', '', 'const a = new Float64Array(n)')).toEqual([
      '1: banned-ok marker that no flagged line follows',
      '3: a typed array as a value container',
    ])
  })

  it('accepts a marker the formatter moved behind a ternary operator', () => {
    expect(
      problems(
        EP,
        'const a = cond',
        '  ? 1',
        '  : // banned-ok: non-numeric: a list of names',
        '    Array.from({ length: n }, f)',
      ),
    ).toEqual([])
  })
})

describe('hand-built arrays: shown code', () => {
  const shown = (...code: string[]): string[] => ['// region demo', ...code, '// endregion']

  it('knows which lines a page shows', () => {
    const lines = ['const a = 1', '// region demo', 'const b = 2', 'const c = 3 // hide', '// endregion', 'const d = 4']
    expect([...regionLines(lines)]).toEqual([2, 3])
    expect(mayShow('apps/pocs/src/examples/x.ts')).toBe(true)
    expect(mayShow('apps/pocs/src/examples/x.test.ts')).toBe(false)
  })

  it('flags loops, pushes and copies in shown code, wherever the file is', () => {
    const path = 'packages/engine/src/lib/optim/example.ts'
    expect(problems(path, ...shown('for (let i = 0; i < n; i++) xs.push(i)'))).toEqual([
      '2: .push( in shown code',
      '2: a for loop in shown code',
    ])
    expect(problems(path, ...shown('const a = Array.from(toFlat(t))'))).toEqual(['2: Array.from( in shown code'])
    expect(problems(path, 'for (let i = 0; i < n; i++) xs.push(i)')).toEqual([])
  })

  it('lets a hidden line through the shown rules, and wants the marker of a shown line hidden', () => {
    const path = 'packages/engine/src/lib/optim/example.ts'
    expect(problems(path, ...shown('const a = Array.from(toFlat(t)) // hide'))).toEqual([])
    expect(
      problems(
        'packages/engine/src/lib/probprog/models/example.ts',
        ...shown('// banned-ok: non-numeric: a row per reading', 'const rows = Array.from({ length: n }, f)'),
      ),
    ).toEqual(['2: a banned-ok marker inside shown code must end with // hide'])
    expect(
      problems(
        'packages/engine/src/lib/probprog/models/example.ts',
        ...shown('// banned-ok: non-numeric: a row per reading // hide', 'const rows = Array.from({ length: n }, f)'),
      ),
    ).toEqual([])
  })
})
