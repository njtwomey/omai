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

/** The box a piece of TeX takes once typeset: its width and its height with its depth, in pixels. */
export interface TexExtent {
  readonly width: number
  readonly height: number
}

/** A node of KaTeX's tree of spans and glyphs, as far as measuring needs it. */
interface KatexDom {
  readonly classes?: readonly string[]
  readonly style?: Readonly<Record<string, string | undefined>>
  readonly children?: readonly KatexDom[]
  readonly text?: string
  readonly width?: number
  readonly italic?: number
  readonly height?: number
  readonly depth?: number
}

type DomTreeBuilder = (tex: string, options: Record<string, unknown>) => KatexDom

/** KaTeX's own sizes (`size1` to `size11`), in ems of the base size: what `\tiny` to `\Huge` and the scripts scale by. */
const KATEX_SIZES = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.44, 1.728, 2.074, 2.488]

/** An `em` length of a style, in ems (anything else counts for nothing). */
function emLength(value: string | undefined): number {
  if (value === undefined || !value.endsWith('em')) return 0
  const v = Number(value.slice(0, -2))
  return Number.isFinite(v) ? v : 0
}

/**
 * The TeX that sets a glyph in the font a symbol node was set in, read from its classes: a run of glyphs KaTeX merged
 * into one node carries only the first one's width, so the others are measured one by one in the same font.
 */
function fontWrap(classes: readonly string[]): readonly [string, string] {
  const has = (c: string): boolean => classes.includes(c)
  if (has('textsf') && has('textbf')) return ['\\textsf{\\textbf{', '}}']
  for (const name of [
    'textbf',
    'textsf',
    'textit',
    'texttt',
    'mathbf',
    'mathit',
    'mathsf',
    'mathtt',
    'mathcal',
    'mathscr',
  ])
    if (has(name)) return [`\\${name}{`, '}']
  if (has('mathnormal')) return ['', '']
  return ['\\mathrm{', '}']
}

const glyphs = new Map<string, number>()

/** The width in ems of one glyph in the font of the given classes (measured through KaTeX once, then cached). */
function glyphWidth(classes: readonly string[], ch: string, build: DomTreeBuilder): number {
  if (ch === ' ') return 0.25
  if (!/^[a-zA-Z0-9.,:;!?()+=<>/|'*@-]$/.test(ch)) return 0.5
  const [open, close] = fontWrap(classes)
  const key = `${open}${ch}${close}`
  const hit = glyphs.get(key)
  if (hit !== undefined) return hit
  const find = (n: KatexDom): number | undefined => {
    if (n.text !== undefined) return n.width
    for (const c of n.children ?? []) {
      const w = find(c)
      if (w !== undefined) return w
    }
    return undefined
  }
  let w = 0.5
  try {
    w = find(build(key, { output: 'html', throwOnError: true, strict: 'ignore' })) ?? 0.5
  } catch {
    w = 0.5
  }
  glyphs.set(key, w)
  return w
}

/** The width of a span or glyph in ems of the size it is drawn at; `scale` is the em of its parent in ems of the base. */
function domWidth(node: KatexDom, scale: number, build: DomTreeBuilder): number {
  if (node.text !== undefined) {
    const classes = node.classes ?? []
    const run = [...node.text]
    const sum = run.length <= 1 ? (node.width ?? 0) : run.reduce((a, ch) => a + glyphWidth(classes, ch, build), 0)
    return (sum + (node.italic ?? 0)) * scale
  }
  const classes = node.classes ?? []
  let own = scale
  if (classes.includes('sizing') || classes.includes('fontsize-ensurer')) {
    const from = classes.find((c) => /^reset-size\d+$/.test(c))
    const to = classes.find((c) => /^size\d+$/.test(c))
    const a = KATEX_SIZES[Number(from?.slice(10)) - 1]
    const b = KATEX_SIZES[Number(to?.slice(4)) - 1]
    if (a !== undefined && b !== undefined) own = scale * (b / a)
  }
  const style = node.style ?? {}
  const edges = emLength(style['marginLeft']) + emLength(style['marginRight'])
  const padding = emLength(style['paddingLeft']) + emLength(style['paddingRight'])
  const widths = (node.children ?? []).map((c) => domWidth(c, own, build))
  // A vertical list is a table: its rows (`vlist-t`) take the widest row, a row's cells (`vlist-r`) add up, and the
  // lines inside a cell (`vlist`) are blocks that take the widest line.
  const inner =
    classes.includes('vlist') || classes.includes('vlist-t')
      ? Math.max(0, ...widths)
      : classes.includes('nulldelimiter')
        ? 0.12 * own
        : classes.includes('strut')
          ? 0
          : widths.reduce((a, b) => a + b, 0)
  const least = emLength(style['minWidth']) * own
  return Math.max(inner, least) + (edges + padding) * own
}

/** KaTeX's stylesheet sets `.katex` to 1.21 times the surrounding font size. */
const KATEX_SCALE = 1.21

const extents = new Map<string, TexExtent>()

/** The cruder estimate used when KaTeX cannot give a tree: commands count one glyph, braces and `^`, `_` nothing. */
function sourceExtent(tex: string): TexExtent {
  // In ems of the base size, like the tree's measure.
  const glyphs = tex
    .replace(
      /\\(?:mathbf|mathrm|mathit|mathsf|mathcal|mathbb|boldsymbol|textbf|textrm|textit|textsf|text|left|right|mid)\b/g,
      '',
    )
    .replace(/\\[a-zA-Z]+/g, 'a')
    .replace(/[{}^_\s\\]/g, '').length
  return { width: glyphs * 0.55, height: 1.2 }
}

/**
 * The size of typeset TeX in pixels at a font size, measured from KaTeX's own layout (every glyph's width in its font
 * and the spacing, scripts, fractions and roots round it), so a node or a label can be given a box that fits the
 * formula rather than its source text: `\mathbf{x}_t` is as wide as a bold x with a small t, `q(x_t \mid x_0)` about
 * five ems. The widths are of KaTeX's HTML output, which the page draws (with the stylesheet that `omai-render/react`
 * imports, which sets it 1.21 times the surrounding size); the result is within a few percent of the browser's own
 * measurement. TeX that does not parse, or a KaTeX
 * without its tree builder, falls back to an estimate from the source (commands count one glyph, braces and `^`, `_`
 * none). Results are cached per formula and macro set.
 *
 * @param tex - The TeX source, without `$` delimiters.
 * @param size - The font size in pixels.
 * @param macros - The macros (default the shared registry).
 * @returns The width, and the height with the depth, in pixels.
 * @example
 * ```ts
 * import { texExtent } from 'omai-render/maths'
 * texExtent('x', 10).width > 0 // => true
 * texExtent('x^2 + y^2', 10).width > texExtent('x^2', 10).width // => true
 * texExtent('\\frac{a}{b}', 20).height > texExtent('a', 20).height // => true
 * ```
 */
export function texExtent(tex: string, size: number, macros?: Readonly<Record<string, string>>): TexExtent {
  const key = tex
  const hit = macros === undefined ? extents.get(key) : undefined
  let em = hit
  if (em === undefined) {
    const build = (katex as unknown as { __renderToDomTree?: DomTreeBuilder }).__renderToDomTree
    try {
      if (build === undefined) throw new Error('no tree builder')
      const tree = build(tex, {
        output: 'html',
        throwOnError: true,
        strict: 'ignore',
        macros: { ...(macros ?? sharedMacros) },
      })
      em = { width: domWidth(tree, 1, build), height: (tree.height ?? 0) + (tree.depth ?? 0) }
      if (!(em.width > 0)) em = sourceExtent(tex)
    } catch {
      em = sourceExtent(tex)
    }
    if (macros === undefined) {
      if (extents.size > 2000) extents.clear()
      extents.set(key, em)
    }
  }
  return { width: em.width * size * KATEX_SCALE, height: em.height * size * KATEX_SCALE }
}
