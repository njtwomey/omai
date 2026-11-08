/**
 * Typesetting TeX with KaTeX and the shared macros, as HTML, MathML or both: `texToHtml`, a caching renderer, a parse
 * check and the compact marker shown in place of TeX that does not parse.
 */

import katex from 'katex'
import { macros as sharedMacros } from './macros.ts'

/** How TeX is typeset: KaTeX's HTML (needs KaTeX's stylesheet on the page), MathML (native in browsers), or both. */
export type TexOutput = 'html' | 'mathml' | 'htmlAndMathml'

/** Options of {@link texToHtml}. */
export interface TexOptions {
  /** A centred block rather than inline maths (default false). */
  readonly display?: boolean
  /** The macros (default the shared registry). */
  readonly macros?: Readonly<Record<string, string>>
  /**
   * Output markup (default `mathml`, which browsers draw without a stylesheet; use `html` or `htmlAndMathml` when
   * the page loads `katex/dist/katex.min.css`).
   */
  readonly output?: TexOutput
}

/**
 * Typeset TeX with KaTeX and the shared macros. Invalid input never throws and never spills raw TeX across the page:
 * it typesets as a compact marker (`⚠ TeX`, class `wb-tex-error`) whose title holds KaTeX's message. `\htmlClass` is
 * allowed (live equations use it to highlight values).
 *
 * @param tex - The TeX source.
 * @param options - Display mode, macros and output markup.
 * @returns The markup.
 * @index function Typeset TeX with KaTeX and shared macros: `texToHtml('x^2')`.
 * @example
 * ```ts
 * import { texToHtml } from 'omai-render/maths'
 * texToHtml('\\xvec^\\top \\yvec').includes('<math') // => true
 * texToHtml('\\frac{1}').includes('wb-tex-error') // => true
 * ```
 */
export function texToHtml(tex: string, options: TexOptions = {}): string {
  try {
    return katex.renderToString(tex, {
      displayMode: options.display ?? false,
      output: options.output ?? 'mathml',
      throwOnError: true,
      strict: 'ignore',
      trust: (context) => context.command === '\\htmlClass',
      // KaTeX may add definitions to the macros object it is given, so it gets a copy.
      macros: { ...(options.macros ?? sharedMacros) },
    })
  } catch (e) {
    return texErrorMarker(e instanceof Error ? e.message : String(e))
  }
}

const escapeHtml = (s: string): string =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')

/**
 * The compact marker that stands in for TeX that does not parse: a warning sign with KaTeX's message as its title.
 *
 * @param message - The parse error.
 * @returns The markup.
 */
export function texErrorMarker(message: string): string {
  return `<span class="wb-tex-error" role="img" aria-label="TeX error" title="${escapeHtml(message)}">⚠ TeX</span>`
}

/**
 * A typesetter with a cache: the same TeX (and display mode) is typeset once. Make one per macro set; the library
 * keeps no global cache.
 *
 * @param options - Macros and output markup for every call.
 * @returns A function from TeX and display mode to markup.
 * @example
 * ```ts
 * import { texRenderer } from 'omai-render/maths'
 * const render = texRenderer()
 * render('x^2') === render('x^2') // => true
 * ```
 */
export function texRenderer(options: Omit<TexOptions, 'display'> = {}): (tex: string, display?: boolean) => string {
  const cache = new Map<string, string>()
  return (tex: string, display = false): string => {
    const key = `${display ? 'D' : 'I'}:${tex}`
    const hit = cache.get(key)
    if (hit !== undefined) return hit
    const html = texToHtml(tex, { ...options, display })
    if (cache.size > 2000) cache.clear()
    cache.set(key, html)
    return html
  }
}

/**
 * Whether TeX parses (with the shared macros), and the parse error if not.
 *
 * @param tex - The TeX source.
 * @param macros - The macros (default the shared registry).
 * @returns `undefined` when it parses, else the error message.
 * @example
 * ```ts
 * import { texError } from 'omai-render/maths'
 * texError('\\frac{1}{2}') // => undefined
 * texError('\\frac{1}').startsWith('KaTeX parse error') // => true
 * ```
 */
export function texError(tex: string, macros: Readonly<Record<string, string>> = sharedMacros): string | undefined {
  try {
    katex.renderToString(tex, { throwOnError: true, strict: 'ignore', macros: { ...macros }, output: 'mathml' })
    return undefined
  } catch (e) {
    return e instanceof Error ? e.message : String(e)
  }
}
