/**
 * Typeset maths in React: `Tex` for TeX with the shared macros, `Equation` for a live equation re-typeset as its values
 * change, `Derivation` for aligned steps with notes, and `MathProvider` for macros and the output markup. KaTeX's
 * stylesheet and fonts are imported here, so the consumer's bundler styles the HTML output with no setup.
 */

import 'katex/dist/katex.min.css'
import './maths.css'
import { type CSSProperties, createContext, type ReactElement, type ReactNode, useContext, useMemo } from 'react'
import { type DerivationStep, derivationTex } from '../maths/derivations.ts'
import { type EquationTemplate, toTex } from '../maths/live.ts'
import { macros as sharedMacros } from '../maths/macros.ts'
import { type TexOutput, texToHtml } from '../maths/tex.ts'

interface MathSettings {
  readonly macros: Readonly<Record<string, string>>
  readonly output: TexOutput
}

const MathContext = createContext<MathSettings>({ macros: sharedMacros, output: 'htmlAndMathml' })

/**
 * Supply macros (merged over the enclosing ones) and the output markup to the maths below.
 *
 * @param props - Extra macros, the output (`htmlAndMathml` by default: KaTeX's HTML, styled by its stylesheet, which this module imports, with MathML for assistive technology), and
 *   the children.
 * @returns The provider.
 */
export function MathProvider(props: {
  readonly macros?: Readonly<Record<string, string>>
  readonly output?: TexOutput
  readonly children?: ReactNode
}): ReactElement {
  const parent = useContext(MathContext)
  const value = useMemo(
    () => ({ macros: { ...parent.macros, ...props.macros }, output: props.output ?? parent.output }),
    [parent, props.macros, props.output],
  )
  return <MathContext.Provider value={value}>{props.children}</MathContext.Provider>
}

/**
 * TeX typeset by KaTeX with the shared macros: `<Tex>{String.raw`\xvec^\top \thetavec`}</Tex>`. Invalid input shows
 * KaTeX's error rather than throwing.
 *
 * @param props - The TeX as children, display mode, extra macros, class and style.
 * @returns The typeset maths.
 */
export function Tex(props: {
  readonly children: string
  readonly display?: boolean
  readonly macros?: Readonly<Record<string, string>>
  readonly className?: string
  readonly style?: CSSProperties
}): ReactElement {
  const settings = useContext(MathContext)
  const { children, display = false, macros } = props
  const html = useMemo(
    () =>
      texToHtml(children, {
        display,
        output: settings.output,
        macros: macros === undefined ? settings.macros : { ...settings.macros, ...macros },
      }),
    [children, display, macros, settings],
  )
  const Tag = display ? 'div' : 'span'
  // biome-ignore lint/security/noDangerouslySetInnerHtml: KaTeX output, from TeX with only \htmlClass trusted
  return <Tag className={props.className} style={props.style} dangerouslySetInnerHTML={{ __html: html }} />
}

/**
 * A live equation: TeX from the `tex` tag (or plain TeX) with its values substituted, re-typeset as they change, so a
 * slider moves the equation and the figure together.
 *
 * @param props - The equation, display mode (default true), class and style.
 * @returns The typeset equation.
 */
export function Equation(props: {
  readonly children: EquationTemplate | string
  readonly display?: boolean
  readonly className?: string
  readonly style?: CSSProperties
}): ReactElement {
  const source = toTex(props.children)
  return (
    <Tex
      display={props.display ?? true}
      {...(props.className === undefined ? {} : { className: props.className })}
      {...(props.style === undefined ? {} : { style: props.style })}
    >
      {source}
    </Tex>
  )
}

/**
 * Steps of a derivation as aligned equations (an `aligned` environment, one step per row, each with an optional note).
 *
 * @param props - The steps (see `derivationTex`) and a class.
 * @returns The typeset derivation.
 */
export function Derivation(props: {
  readonly steps: readonly DerivationStep[]
  readonly className?: string
}): ReactElement {
  return (
    <Tex display {...(props.className === undefined ? {} : { className: props.className })}>
      {derivationTex(props.steps)}
    </Tex>
  )
}
