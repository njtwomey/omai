import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { isOmaiError } from '../base/index.ts'
import {
  crfppFeatures,
  crfppWeightOffsets,
  expandAttributeTemplates,
  featureIndex,
  fireFeatures,
  offsetConjunctions,
  parseAttributeTemplates,
  parseCrfppTemplate,
  parseWapitiPatterns,
  wapitiFeatures,
} from './index.ts'

type Part = { readonly lit: string } | { readonly row: number; readonly col: number }

const literal = fc.string({ unit: fc.constantFrom('a', 'b', ':', '/', '|', '_', '0', '1', '='), maxLength: 4 })
const part: fc.Arbitrary<Part> = fc.oneof(
  literal.map((lit) => ({ lit })),
  fc.record({ row: fc.integer({ min: -8, max: 8 }), col: fc.integer({ min: 0, max: 2 }) }),
)
const line = fc.record({ kind: fc.constantFrom('U', 'B'), id: fc.nat(99), parts: fc.array(part, { maxLength: 4 }) })
const table = fc.array(
  fc.tuple(fc.constantFrom('the', 'cat', 'sat', 'Ünï'), fc.constantFrom('DT', 'NN'), fc.constantFrom('x', 'y')),
  {
    minLength: 1,
    maxLength: 6,
  },
)

const render = (l: { kind: string; id: number; parts: readonly Part[] }): string =>
  `${l.kind}${String(l.id).padStart(2, '0')}:${l.parts.map((p) => ('lit' in p ? p.lit : `%x[${p.row},${p.col}]`)).join('')}`

/** A direct reading of CRF++'s rule: the cell at position + row, or _B-k / _B+k outside. */
function reference(parts: readonly Part[], prefix: string, rows: readonly (readonly string[])[], t: number): string {
  return (
    prefix +
    parts
      .map((p) => {
        if ('lit' in p) return p.lit
        const i = t + p.row
        if (i < 0) return `_B-${-i}`
        if (i >= rows.length) return `_B+${i + 1 - rows.length}`
        return rows[i]?.[p.col] ?? ''
      })
      .join('')
  )
}

describe('CRF++ templates', () => {
  it('expansion agrees with a direct reading of the macro rule at every position', () => {
    fc.assert(
      fc.property(fc.array(line, { maxLength: 5 }), table, (lines, rows) => {
        const t = parseCrfppTemplate(lines.map(render).join('\n'))
        const f = crfppFeatures(t, rows)
        const us = lines.filter((l) => l.kind === 'U')
        const bs = lines.filter((l) => l.kind === 'B')
        for (let p = 0; p < rows.length; p++) {
          expect(f.unigram[p]).toEqual(us.map((l) => reference(l.parts, `U${String(l.id).padStart(2, '0')}:`, rows, p)))
          expect(f.bigram[p]).toEqual(
            p === 0 ? [] : bs.map((l) => reference(l.parts, `B${String(l.id).padStart(2, '0')}:`, rows, p)),
          )
          expect(fireFeatures(t, rows, p).map((x) => x.feature)).toEqual([
            ...(f.unigram[p] ?? []),
            ...(f.bigram[p] ?? []),
          ])
        }
      }),
    )
  })

  it('reports syntax errors with line and column', () => {
    const cases: [string, number, number][] = [
      ['U00:%x[0,0]\nX01:a', 2, 1],
      ['U00:%y[0,0]', 1, 5],
      ['U00:a%x[0 ,0]', 1, 6],
      ['\n\nB:%x[9,0]', 3, 3],
    ]
    for (const [src, l, c] of cases) {
      try {
        parseCrfppTemplate(src)
        expect.unreachable()
      } catch (e) {
        expect(isOmaiError(e) && e.details['line'] === l && e.details['column'] === c).toBe(true)
      }
    }
    expect(() => crfppFeatures(parseCrfppTemplate('U:%x[0,3]'), [['a']])).toThrow(/column 3/)
  })

  it('skips comments, blank lines and lines starting with a space; keeps empty macro digits as 0', () => {
    const t = parseCrfppTemplate('# c\n \tU99:x\n\r\nU0:%x[,]\r\nB')
    expect(t.unigram.map((l) => l.source)).toEqual(['U0:%x[,]'])
    expect(t.unigram[0]?.parts).toEqual(['U0:', { row: 0, col: 0 }])
    expect(t.bigram.length).toBe(1)
  })

  it('feature indexes respect the frequency cut-off and CRF++ weight offsets', () => {
    fc.assert(
      fc.property(
        fc.array(fc.array(fc.array(fc.constantFrom('U1:a', 'U1:b', 'B', 'B2:c'), { maxLength: 3 }), { maxLength: 4 }), {
          maxLength: 4,
        }),
        fc.integer({ min: 1, max: 3 }),
        (seqs, min) => {
          const idx = featureIndex(seqs, { minFrequency: min })
          const counts = new Map<string, number>()
          for (const s of seqs) for (const item of s) for (const f of item) counts.set(f, (counts.get(f) ?? 0) + 1)
          expect(idx.features).toEqual([...counts].filter(([, c]) => c >= min).map(([f]) => f))
          const { offsets, size } = crfppWeightOffsets(idx, 3)
          let total = 0
          for (const f of idx.features) {
            expect(offsets.get(f)).toBe(total)
            total += f.startsWith('B') ? 9 : 3
          }
          expect(size).toBe(total)
        },
      ),
    )
  })
})

describe('Wapiti patterns', () => {
  it('expand extraction, tests and matches, with boundaries, case-insensitivity and absolute offsets', () => {
    const p = parseWapitiPatterns(
      'u:w=%x[-5,0]|%x[-1,0]|%x[2,0]\nb:%T[0,0,"^the$"]\n*:%m[0,1,"\\u+"]\nU:a=%x[@0,0],%x[@-1,0]',
    )
    const f = wapitiFeatures(p, [
      ['The', 'DT'],
      ['cat', 'NN'],
    ])
    expect(f.unigram).toEqual([
      ['u:w=_x-#|_x-1|_x+1', '*:DT', 'U:a=The,cat'],
      ['u:w=_x-4|The|_x+2', '*:NN', 'U:a=The,cat'],
    ])
    expect(f.bigram).toEqual([[], ['b:false', '*:NN']])
    expect(wapitiFeatures(parseWapitiPatterns('b:%t[0,0,"^the$"]'), [['the']]).bigram).toEqual([[]])
    expect(() => parseWapitiPatterns('u:%x[0,0,"a"]')).toThrow(/takes no pattern/)
    expect(() => parseWapitiPatterns('q:abc')).toThrow(/unknown pattern type/)
  })
})

describe('attribute templates and offset conjunctions', () => {
  it('attribute templates skip or pad outside the sequence and parse from text', () => {
    const tokens = [{ w: 'a' }, { w: 'b' }, { w: 'c' }]
    const templates = parseAttributeTemplates('w[0]\nw[-1]|w[0]; w[2]')
    expect(templates).toEqual([
      [['w', 0]],
      [
        ['w', -1],
        ['w', 0],
      ],
      [['w', 2]],
    ])
    const skipped = expandAttributeTemplates(tokens, templates, { bosEos: true })
    expect(skipped[0]).toEqual({ 'w[0]=a': 1, 'w[2]=c': 1, __BOS__: 1 })
    expect(skipped[2]).toEqual({ 'w[0]=c': 1, 'w[-1]|w[0]=b|c': 1, __EOS__: 1 })
    const padded = expandAttributeTemplates(tokens, templates, { boundary: 'pad' })
    expect(padded[2]?.['w[2]=_B+2']).toBe(1)
    expect(() => parseAttributeTemplates('w[x]')).toThrow(/malformed/)
  })

  it('offset conjunctions match a brute-force product of features', () => {
    const items = [{ 'W=a': 1, 'S=x': 2 }, { 'W=b': 1 }, { 'W=c': 0.5, 'S=y': 1 }]
    const out = offsetConjunctions(items, [[-1], [1], [-1, 0], [0, 0], [-2, -1, 0]])
    expect(out[1]).toEqual({
      'W=b': 1,
      'W=a@-1': 1,
      'S=x@-1': 2,
      'W=c@1': 0.5,
      'S=y@1': 1,
      'W=a@-1&W=b': 1,
      'S=x@-1&W=b': 2,
    })
    expect(out[0]).toEqual({ 'W=a': 1, 'S=x': 2, 'W=b@1': 1, 'W=a&S=x': 2 })
    expect(out[2]?.['W=a@-2&W=b@-1&S=y']).toBe(1)
    expect(out[2]?.['W=c&S=y']).toBe(0.5)
    expect(() => offsetConjunctions(items, [[0]])).toThrow(/singletons/)
    expect(offsetConjunctions(items, [[0]], { includeOriginalSingletons: false })[0]).toEqual({ 'W=a': 1, 'S=x': 2 })
  })
})
