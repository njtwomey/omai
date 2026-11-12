import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { Choice, type ChoiceOption } from './Choice.tsx'

const OPTIONS: readonly ChoiceOption<string>[] = [
  { value: 'round', label: 'Round', description: 'one spread for both directions' },
  { value: 'full', label: 'Full', description: 'a spread in each direction and a tilt' },
]

describe('the choice group', () => {
  it('is a labelled group of radio inputs, one per option, with the chosen one checked', () => {
    const html = renderToString(<Choice label="Shape" options={OPTIONS} value="full" onChange={() => {}} />)
    expect(html).toContain('<fieldset')
    expect(html).toContain('<legend')
    expect(html).toContain('Shape')
    expect(html.match(/type="radio"/g)).toHaveLength(2)
    // one name for the group, so the arrow keys move between the choices
    const names = [...html.matchAll(/name="([^"]+)"/g)].map((m) => m[1])
    expect(new Set(names).size).toBe(1)
    // only the second input is checked
    const inputs = html.match(/<input[^>]*>/g) ?? []
    expect(inputs[0]).not.toContain('checked')
    expect(inputs[1]).toContain('checked')
  })

  it('shows each option as a label with its description under it, and ties the description to the input', () => {
    const html = renderToString(<Choice label="Shape" options={OPTIONS} value="round" onChange={() => {}} />)
    for (const o of OPTIONS) {
      expect(html).toContain(o.label)
      expect(html).toContain(o.description)
    }
    expect(html.match(/aria-describedby="/g)).toHaveLength(2)
    expect(html.match(/<label/g)).toHaveLength(2)
  })

  it('works with numbers as values', () => {
    const numbers: readonly ChoiceOption<number>[] = [
      { value: 2, label: '2 groups', description: 'two' },
      { value: 3, label: '3 groups', description: 'three' },
    ]
    const html = renderToString(<Choice label="Groups" options={numbers} value={3} onChange={() => {}} />)
    expect(html).toContain('value="3"')
    expect((html.match(/<input[^>]*>/g) ?? [])[1]).toContain('checked')
  })
})
