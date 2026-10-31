import { defaults, dict, int, union } from 'mdml-engine/space'
import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { categoricalColour } from '../../theme/palettes.ts'
import { SlotVariants, variantColours } from './SlotVariants.tsx'

const solvers = union(
  { a: dict({ k: int(1, 5, { default: 2 }) }), b: dict({}), c: dict({ m: int(1, 9) }) },
  { tag: 'kind' },
)

describe('SlotVariants', () => {
  it('gives the options that apply one colour each in the order the comparison runs them', () => {
    expect(variantColours(['a', 'b', 'c'])).toEqual({
      a: categoricalColour(0),
      b: categoricalColour(1),
      c: categoricalColour(2),
    })
    expect(variantColours(['a', 'b', 'c'], (n) => n !== 'b')).toEqual({
      a: categoricalColour(0),
      c: categoricalColour(1),
    })
  })

  it('lists every option with a settings button on those with knobs, greyed with the reason when ruled out', () => {
    const html = renderToString(
      <SlotVariants
        space={solvers}
        knobs={{ a: defaults(solvers.options['a'] as never) as never }}
        onChange={() => undefined}
        label="solver"
        admissible={(n) => n !== 'c'}
        why={(n) => `${n} needs a prox`}
      />,
    )
    expect(html.match(/data-variant=/g)).toHaveLength(3)
    expect(html).toContain('aria-label="Settings of a"')
    expect(html).toContain('b has no settings')
    expect(html).toContain('title="c needs a prox"')
    expect(html).toContain('data-variant="c" data-allowed="false"')
  })
})
