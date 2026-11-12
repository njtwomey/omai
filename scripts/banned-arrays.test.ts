import { describe, expect, it } from 'vitest'
import { arrayProblems, mayShow, regionLines } from './banned-arrays.ts'

const SRC = 'packages/engine/src/lib/probprog/ep/_atoms/example.ts'

const shown = (...lines: string[]): string => ['// region demo', ...lines, '// endregion'].join('\n')
const problems = (path: string, text: string): string[] =>
  arrayProblems(path, text).map((p) => `${p.line}: ${p.message.split(' ')[0]}`)

describe('hand-built arrays in shown code', () => {
  it('knows which lines a page shows', () => {
    const lines = ['a', '// region r', 'b', 'c', '// endregion', 'd', '{/* region s */}', 'e', '{/* endregion */}']
    expect([...regionLines(lines)]).toEqual([2, 3, 7])
  })

  it('only looks at source files that can hold shown code, never at tests', () => {
    expect(mayShow(SRC)).toBe(true)
    expect(mayShow('packages/engine/src/lib/probprog/ep/_atoms/example.test.ts')).toBe(false)
    expect(mayShow('scripts/example.ts')).toBe(false)
  })

  it.each([
    ['new Array', 'const a = new Array<number>(n).fill(0)'],
    ['.fill(', 'row.fill(1)'],
    ['Array.from(', 'const a = Array.from({ length: n }, (_, i) => i)'],
    ['Array.from( of a tensor', 'const a = Array.from(toFlat(t))'],
    ['a typed array', 'const buffer = new Float64Array(n)'],
    ['.push(', 'out.push(x)'],
    ['a for loop', 'for (let i = 0; i < n; i++) total += x[i]'],
  ])('flags %s', (_, code) => {
    expect(arrayProblems(SRC, shown(code)).length).toBeGreaterThan(0)
  })

  it('passes tensor code, comments and strings', () => {
    expect(
      arrayProblems(
        SRC,
        shown(
          'const a = full([n], 0)',
          'const b = toFlat(arange(n))',
          "const c = 'new Array(n).fill(0)'",
          '// new Float64Array(n) in a comment',
          'const d = tensor([1, 2, 3])',
        ),
      ),
    ).toEqual([])
  })

  it('does not look outside a region: the rule there is the one in CLAUDE.md', () => {
    expect(arrayProblems(SRC, 'const a = new Array<number>(n).fill(0)\nfor (const x of xs) out.push(x)')).toEqual([])
  })

  it('lets a hidden line through: the page does not show it', () => {
    expect(problems(SRC, shown('const a = new Array<number>(n).fill(0) // hide'))).toEqual([])
  })

  it('reports the line a page would show', () => {
    expect(problems(SRC, shown('const ok = zeros([n])', 'const bad = new Array(n)'))).toEqual(['3: new'])
  })
})
