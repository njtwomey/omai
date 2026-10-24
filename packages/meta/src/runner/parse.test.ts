import { describe, expect, it } from 'vitest'
import { bracketBalance, importsOf, parseExample, splitComment } from './parse.ts'
import { instrument } from './transform.ts'

describe('splitComment', () => {
  it('splits at the first // outside a string', () => {
    expect(splitComment("f('a // b') // => 1")).toEqual({ code: "f('a // b')", comment: ' => 1' })
    expect(splitComment('const s = `x//y`')).toEqual({ code: 'const s = `x//y`', comment: null })
    expect(splitComment("'it\\'s' // c")).toEqual({ code: "'it\\'s'", comment: ' c' })
  })
})

describe('bracketBalance', () => {
  it('counts brackets outside strings', () => {
    expect(bracketBalance('f(a, [1, {')).toBe(3)
    expect(bracketBalance("')' + g()")).toBe(0)
    expect(bracketBalance('}])')).toBe(-3)
  })
})

describe('importsOf', () => {
  it('lists static import specifiers once', () => {
    const code = "import { a } from 'x'\nimport type { T } from 'y'\nimport * as z from 'x'\nimport 'side'"
    expect(importsOf(code)).toEqual(['x', 'y', 'side'])
  })
})

describe('parseExample', () => {
  it('finds value, throws, figure and trace markers', () => {
    const p = parseExample('f(1) // => 2\ng() // => throws shape\nh() // => figure\nrun() // => trace')
    expect(p.expectations.map((e) => [e.kind, e.expression, e.expected])).toEqual([
      ['value', 'f(1)', '2'],
      ['throws', 'g()', 'shape'],
      ['figure', 'h()', ''],
      ['trace', 'run()', ''],
    ])
  })

  it('walks back over a multi-line expression until the brackets balance', () => {
    const p = parseExample('const a = 1\nf(\n  a,\n) // => 3')
    expect(p.expectations[0]).toMatchObject({ startLine: 1, line: 3, expression: 'f(\n  a,\n)' })
  })

  it('takes in chained calls and unfinished lines above', () => {
    const p = parseExample('xs\n  .map(f)\n  .length // => 2\nconst y =\n  g() // => 1')
    expect(p.expectations[0]).toMatchObject({ startLine: 0, line: 2 })
    expect(p.expectations[1]).toMatchObject({ startLine: 3, line: 4, prefix: 'const y =' })
  })

  it('applies a marker on its own line to the expression above', () => {
    const p = parseExample("format(k)\n// => 'k'")
    expect(p.expectations[0]).toMatchObject({ line: 0, markerLine: 1, expression: 'format(k)', expected: "'k'" })
  })

  it('keeps a declaration outside the instrumented expression', () => {
    const p = parseExample('const x: number = f() // => 1')
    expect(p.expectations[0]).toMatchObject({ prefix: 'const x: number = ', expression: 'f()' })
  })

  it('instruments without moving lines', () => {
    const p = parseExample('a\n;(b) // => 1\nc() // => throws\nd\n// => figure')
    const out = instrument(p).split('\n')
    expect(out).toEqual(['a', ';__at(0, ((b)), () => (1))', 'await __throws(1, async () => (c()))', '__at(2, (d))', ''])
  })
})
