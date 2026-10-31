import { describe, expect, it } from 'vitest'
import { framesBetween, pageSvg, probeRequestsOf, wrapText } from './export.ts'
import type { Narrative, Panel } from './layout.ts'

describe('probeRequestsOf', () => {
  it('lists the derived paths a layout reads, each once, with arguments', () => {
    const layout: Panel[] = [
      { kind: 'curves', paths: ['loss'] },
      { kind: 'probe', path: 'plane/distance' },
      { kind: 'probe', path: 'plane/distance' },
      { kind: 'probe', path: 'plane/distance', args: { n: 3 } },
      { kind: 'field', path: 'flow' },
      { kind: 'tensor', path: 'm', marksPath: 'marks', marksArgs: { k: 1 } },
      { kind: 'trajectory', paths: ['p'], over: { path: 'surface' } },
      { kind: 'rows', overlay: 'surface', values: { path: 'scores' } },
      { kind: 'tape' },
    ]
    expect(probeRequestsOf(layout)).toEqual([
      { path: 'plane/distance' },
      { path: 'plane/distance', args: { n: 3 } },
      { path: 'flow' },
      { path: 'marks', args: { k: 1 } },
      { path: 'surface' },
      { path: 'model/decision-surface' },
      { path: 'scores' },
    ])
  })

  it('reads every page of a narrative', () => {
    const narrative: Narrative = {
      pages: [
        { id: 'a', title: 'a', text: '', layout: [{ kind: 'probe', path: 'x' }] },
        {
          id: 'b',
          title: 'b',
          text: '',
          layout: [
            { kind: 'probe', path: 'y' },
            { kind: 'probe', path: 'x' },
          ],
        },
      ],
    }
    expect(probeRequestsOf(narrative).map((r) => r.path)).toEqual(['x', 'y'])
  })
})

describe('framesBetween', () => {
  it('takes every step when few, an even spread with both ends when many, and reads a backward range forwards', () => {
    expect(framesBetween(2, 5, 10)).toEqual([2, 3, 4, 5])
    expect(framesBetween(5, 2, 10)).toEqual([2, 3, 4, 5])
    const many = framesBetween(0, 1000, 11)
    expect(many).toHaveLength(11)
    expect(many[0]).toBe(0)
    expect(many.at(-1)).toBe(1000)
    expect(many[5]).toBe(500)
  })
})

describe('the page frame', () => {
  it('wraps a caption to the width', () => {
    expect(wrapText('one two three four', 9)).toEqual(['one two', 'three', 'four'])
    expect(wrapText('', 9)).toEqual([])
  })

  it('puts the figures under the title and over the caption on the page background', () => {
    const page = pageSvg('<svg width="400" height="200"></svg>', 400, 200, {
      title: 'A & B',
      caption: 'a caption that is long enough to wrap onto a second line when the page is narrow, as it is',
      mode: 'dark',
    })
    expect(page.width).toBe(400)
    expect(page.height).toBeGreaterThan(244)
    expect(page.svg).toContain('A &amp; B')
    expect(page.svg).toContain('translate(0 44)')
    expect(page.svg.match(/<text /g)?.length).toBeGreaterThan(2)
    const plain = pageSvg('<svg/>', 400, 200, { mode: 'light' })
    expect(plain.height).toBe(200)
  })
})
