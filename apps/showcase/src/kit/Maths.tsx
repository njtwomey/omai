import { type EquationTemplate, toTex } from 'omai-render/maths'
import { Tex as TexView } from 'omai-render/react'
import type { ReactNode } from 'react'

export { live, tex } from 'omai-render/maths'

/**
 * A value read from the result, set in TeX as a highlighted slot (the way the `tex` tag does for its slots), for a formula built
 * from parts with `String.raw`: ``String.raw`y_{${j}} = ${slot(fixed(v))}` ``. Only values from the result go in slots; constants
 * of the model are written in the TeX as they are.
 *
 * @param value - The value, as the text the story's formatter made of it.
 * @returns The TeX for the slot.
 * @example
 * ```ts
 * slot('2.10') // => '\\htmlClass{wb-slot}{2.10}'
 * ```
 */
export function slot(value: string | number): string {
  return `\\htmlClass{wb-slot}{${value}}`
}

/** What a piece of maths is written as: TeX, or the `tex` tag with the numbers read from a result in its slots. */
export type TexSource = string | EquationTemplate

/**
 * The text a screen reader gets in place of typeset maths: the plain-language `reading` when the author wrote one, else the TeX
 * source with the markup of live values (`\htmlClass{wb-slot}{2.1}`) reduced to the value.
 *
 * @param source - The TeX, or the template.
 * @param reading - The author's reading in words.
 * @returns The text alternative.
 * @example
 * ```ts
 * alternativeOf('J = \\htmlClass{wb-slot}{2}') // => 'J = 2'
 * alternativeOf('J', 'the precision') // => 'the precision'
 * ```
 */
export function alternativeOf(source: TexSource, reading?: string): string {
  if (reading !== undefined) return reading
  return toTex(source).replace(/\\htmlClass\{[^}]*\}\{([^{}]*)\}/g, '$1')
}

/**
 * Inline maths in a sentence, typeset by omai-render with its shared macros (`\Gauss`, `\xvec`, `\expect`, `\KL`): the symbol of a
 * quantity (`J`, `\mu`, `\sigma^2`) or a short formula. Put the numbers read from the result in the slots of the `tex` tag, so
 * they are typeset in the formula and are still read from the result:
 * `<Tex>{tex`J = ${fixed(precision)}`}</Tex>`. It is not for code names, which stay in `<code>`.
 *
 * @param props - The TeX (written with `String.raw` or the `tex` tag, so backslashes need no doubling) and an optional reading in
 *   words for screen readers (the default is the TeX).
 * @returns The maths, in a span.
 */
export function Tex(props: { readonly children: TexSource; readonly reading?: string }): ReactNode {
  const source = toTex(props.children)
  return (
    <span role="math" aria-label={alternativeOf(props.children, props.reading)} className="story-tex">
      <TexView>{source}</TexView>
    </span>
  )
}

/**
 * A display equation: centred on its own line, in the column of the text, with the same spacing as a figure and an optional short
 * label under it. Use it after the sentence that says the same thing in words, so the words come first and the formula second.
 * A line that is too wide for a narrow screen scrolls sideways inside its own box rather than widening the page.
 *
 * @param props - The TeX (a `tex` template for values read from the result), a short label, and an optional reading in words for
 *   screen readers (the default is the TeX).
 * @returns The equation.
 */
export function Equation(props: {
  readonly children: TexSource
  readonly label?: ReactNode
  readonly reading?: string
}): ReactNode {
  const source = toTex(props.children)
  return (
    <figure className="story-block story-equation m-0">
      <div role="math" aria-label={alternativeOf(props.children, props.reading)} className="story-equation-box">
        <TexView display>{source}</TexView>
      </div>
      {props.label === undefined ? null : (
        <figcaption className="mt-1 px-1 text-center font-sans text-sm leading-relaxed text-muted">
          {props.label}
        </figcaption>
      )}
    </figure>
  )
}
