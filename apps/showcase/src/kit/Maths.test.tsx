import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { alternativeOf, Equation, Tex, tex } from './Maths.tsx'

describe('maths in a story', () => {
  it('typesets inline maths with the shared macros, with a text alternative', () => {
    const html = renderToString(
      <p>
        The spread is <Tex>{String.raw`\Gauss(\mu, \sigma^2)`}</Tex>.
      </p>,
    )
    expect(html).toContain('class="story-tex"')
    expect(html).toContain('role="math"')
    expect(html).toContain('katex')
    expect(html).not.toContain('wb-tex-error')
    expect(html).toContain('aria-label="\\Gauss(\\mu, \\sigma^2)"')
  })

  it('puts the values read from a result into a formula, and keeps the markup out of the text alternative', () => {
    const precision = (1 / 4).toFixed(2)
    const html = renderToString(<Tex reading="the precision is 0.25">{tex`J = ${precision}`}</Tex>)
    expect(html).toContain('wb-slot')
    expect(html).toContain('aria-label="the precision is 0.25"')
    expect(alternativeOf(tex`J = ${precision}`)).toBe('J = 0.25')
  })

  it('sets a display equation in the figure spacing, centred, with an optional label', () => {
    const html = renderToString(<Equation label="The cavity">{String.raw`\text{cavity} = q / g_k`}</Equation>)
    expect(html).toContain('story-block story-equation')
    expect(html).toContain('katex-display')
    expect(html).toContain('<figcaption')
    expect(html).toContain('The cavity')
    expect(renderToString(<Equation>{'x'}</Equation>)).not.toContain('<figcaption')
  })

  it('shows a marker, not raw source, for TeX that does not parse', () => {
    expect(renderToString(<Tex>{String.raw`\frac{1}`}</Tex>)).toContain('wb-tex-error')
  })
})
