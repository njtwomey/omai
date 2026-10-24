import { describe, expect, it } from 'vitest'
import { clauseToString, parseClause, parseProgram, parseQuery, parseTerm, termToString, tokenize } from './index.ts'

const show = (s: string): string => termToString(parseTerm(s))
const canon = (s: string): string => termToString(parseTerm(s), { ignoreOps: true })

describe('Prolog reader', () => {
  it('respects operator priorities and associativity', () => {
    expect(canon('1 + 2 * 3')).toBe('+(1,*(2,3))')
    expect(canon('1 - 2 - 3')).toBe('-(-(1,2),3)')
    expect(canon('2 ^ 3 ^ 4')).toBe('^(2,^(3,4))')
    expect(canon('a :- b, c ; d -> e')).toBe(":-(a,;(','(b,c),->(d,e)))")
    expect(canon('\\+ a, b')).toBe("','(\\+(a),b)")
    expect(canon('- 1')).toBe('-(1)')
    expect(canon('-1')).toBe('-1')
    expect(canon('- (1)')).toBe('-(1)')
    expect(canon('a - -1')).toBe('-(a,-1)')
    expect(canon('X = (a :- b)')).toBe('=(X,:-(a,b))')
    expect(canon('f(- , +)')).toBe('f(-,+)')
    expect(canon('[a, b | T]')).toBe('[a,b|T]')
    expect(canon('{a, b}')).toBe("{}(','(a,b))")
    expect(canon('(a | b)')).toBe(';(a,b)')
    expect(canon('p :- dynamic foo/1')).toBe(':-(p,dynamic(/(foo,1)))')
    expect(canon('X is 5 mod 2')).toBe('is(X,mod(5,2))')
  })

  it('reads numbers, quoted atoms, strings and comments', () => {
    expect(canon("0'a + 0x1F + 0b101 + 0o17 + 1.5e3")).toBe('+(+(+(+(97,31),5),15),1500.0)')
    expect(canon("'don''t'")).toBe("'don\\'t'")
    expect(canon("'a\\nb'")).toBe("'a\\nb'")
    expect(canon('"ab"')).toBe('[97,98]')
    expect(termToString(parseTerm('"ab"', { doubleQuotes: 'chars' }))).toBe('[a,b]')
    expect(termToString(parseTerm('"ab"', { doubleQuotes: 'atom' }))).toBe('ab')
    expect(tokenize('a. /* x */ b % y\n.').map((t) => t.type)).toEqual(['atom', 'end', 'atom', 'end'])
  })

  it('reads programs, op directives and queries', () => {
    const p = parseProgram(`
      :- op(700, xfx, ===>).
      rule(X) ===> X.
      likes(mary, wine).
      greeting --> [hello].`)
    expect(p.clauses.map((c) => clauseToString(c, { operators: p.operators }))).toEqual([
      'rule(X)===>X.',
      'likes(mary,wine).',
      'greeting(_DCG0,_DCG1) :- _DCG0=[hello|_DCG1].',
    ])
    expect(parseQuery('X = Y, _ = Z, _A = 1').variables).toEqual(['X', 'Y', 'Z', '_A'])
    expect(parseClause('p :- q, r.').body.length).toBe(2)
  })

  it('reports syntax errors', () => {
    for (const bad of ['f(a', 'a :- .', "'unterminated", 'f(a))', 'a b', '/* open']) {
      expect(() => parseTerm(bad)).toThrow(/syntax error/)
    }
  })

  it('writes atoms, operators and numbers so that they read back', () => {
    expect(show("f('A', [], '[]', {}, 'hello world', ',', '|')")).toBe("f('A',[],[],{},'hello world',',','|')")
    expect(show('- (1)')).toBe('- 1')
    expect(show('-(-(1))')).toBe('- - 1')
    expect(show('1 - (-1)')).toBe('1- -1')
    expect(show('- (a)')).toBe('-a')
    expect(show('\\+ (a, b)')).toBe('\\+ (a,b)')
    expect(show('(a :- b) :- c')).toBe('(a:-b):-c')
    expect(show('f((a, b))')).toBe('f((a,b))')
    expect(show('X is Y mod 2')).toBe('X is Y mod 2')
    expect(show('(dynamic) - 1')).toBe('(dynamic)-1')
    expect(show('- (1 ^ 2)')).toBe('- 1^2')
    expect(show('1.0e10')).toBe('10000000000.0')
  })
})
