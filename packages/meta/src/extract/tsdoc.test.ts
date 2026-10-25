import { describe, expect, it } from 'vitest'
import { parseDocComment, unframeComment } from './tsdoc.ts'

const COMMENT = `/**
 * Elementwise sum with broadcasting.
 * Plain numbers are weak.
 *
 * A second paragraph.
 *
 * @param a - Left operand.
 * @param options.tol - Tolerance,
 *   on two lines.
 * @returns \`a + b\`.
 * @throws MdmlError with code \`shape\`.
 * @concept broadcasting tensor
 * @see numpy.add
 * @reference Golub and Van Loan (2013).
 * @remarks Pure.
 * @example Adding rows
 * \`\`\`ts
 * import { add } from 'mdml-engine/tensor'
 * // @param inside code is not a tag
 * add(1, 2) // => 3
 * \`\`\`
 * @example
 * \`\`\`ts
 * add(0, 0) // => 0
 * \`\`\`
 */`

describe('unframeComment', () => {
  it('keeps one line per comment line', () => {
    expect(unframeComment(COMMENT).split('\n')).toHaveLength(COMMENT.split('\n').length)
  })
})

describe('parseDocComment', () => {
  const doc = parseDocComment(unframeComment(COMMENT))

  it('splits summary, description and tags', () => {
    expect(doc.summary).toBe('Elementwise sum with broadcasting. Plain numbers are weak.')
    expect(doc.description).toBe('A second paragraph.\n\nPure.')
    expect(doc.params).toEqual([
      { name: 'a', description: 'Left operand.' },
      { name: 'options.tol', description: 'Tolerance, on two lines.' },
    ])
    expect(doc.returns).toBe('`a + b`.')
    expect(doc.throws).toEqual(['MdmlError with code `shape`.'])
    expect(doc.concepts).toEqual(['broadcasting', 'tensor'])
    expect(doc.see).toEqual(['numpy.add'])
    expect(doc.references).toEqual(['Golub and Van Loan (2013).'])
  })

  it('reads fenced examples with titles and the line their code starts on', () => {
    expect(doc.examples).toHaveLength(2)
    expect(doc.examples[0]?.title).toBe('Adding rows')
    expect(doc.examples[0]?.code).toBe(
      "import { add } from 'mdml-engine/tensor'\n// @param inside code is not a tag\nadd(1, 2) // => 3",
    )
    const lines = unframeComment(COMMENT).split('\n')
    expect(lines[doc.examples[0]?.source?.line ?? -1]).toBe("import { add } from 'mdml-engine/tensor'")
    expect(doc.examples[1]).toMatchObject({ title: '', code: 'add(0, 0) // => 0' })
  })

  it('collects modifier tags, inline or on their own line', () => {
    const d = parseDocComment('Elementwise sum. @internal')
    expect(d.summary).toBe('Elementwise sum.')
    expect(d.modifiers).toEqual(['internal'])
    expect(parseDocComment('Old.\n\n@deprecated Use `b`.\n@beta').modifiers).toEqual(['beta'])
    expect(parseDocComment('Old.\n\n@deprecated Use `b`.').deprecated).toBe('Use `b`.')
  })

  it('reads unfenced examples', () => {
    expect(parseDocComment('S.\n@example f(1) // => 2').examples[0]).toMatchObject({ title: '', code: 'f(1) // => 2' })
    expect(parseDocComment('S.\n@example Title\nf(1) // => 2').examples[0]).toMatchObject({ title: 'Title' })
  })
})
