/**
 * The theme in React: `ThemeProvider` and `useColourMode` take the mode from a prop or from the page's root `dark`
 * class (dark by default, no system-preference detection), and `useTokens` gives the design tokens of the current mode
 * with the host's overrides merged in.
 *
 * ## The provider
 *
 * `<ThemeProvider mode tokens cssVariables>` is the one source of the theme for everything below it (charts, scenes,
 * frames, views, controls, players) and nests: a slide can pin `mode="light"` inside a dark deck, and the figures,
 * controls and popups inside it follow. It renders one wrapper `div` (`display: contents` by default, so it does not
 * disturb the layout; `className` and `style` are passed to it) that carries `data-theme` (the mode) and the CSS
 * variables below.
 *
 * - `mode`: the colour mode below. Without it the mode is inherited from the enclosing provider, else the page's. A
 *   provider that pins a mode other than the one it sits in also writes every `--wb-*` variable of that mode on its
 *   wrapper, so the HTML controls change with the charts; one that pins the mode already in effect leaves the host's
 *   variables alone.
 * - `tokens`: a partial set of tokens merged over `tokens(mode)`, for a host that restyles charts and scenes: an object
 *   (`{ accent: '#7c3aed' }`), one per mode (`{ light: {...}, dark: {...} }`) or a function of the mode. Nested providers
 *   merge, the inner one over the outer. The overrides reach everything drawn from tokens (`useTokens`, `SceneView`,
 *   `FigureFrame`, the figure views, and `Chart`, whose ECharts theme is made from them: series colours, text, axes,
 *   legend, tooltip) and the HTML controls, through the `--wb-*` variables the wrapper writes for the tokens overridden.
 *   Colours a chart builder wrote into an option from `tokens(mode)` itself are the mode's, not the override's: build the
 *   option with a colour from `useTokens()` to restyle one.
 * - `cssVariables`: a map from a token name to the host's own variable, `{ fg: '--foreground', muted: '--muted-foreground' }`,
 *   written on the wrapper as `--wb-fg: var(--foreground)` and so on (a value that is not a bare `--name` is used as it
 *   is, `'hsl(var(--foreground))'`). It takes the place of the mapping in the host's stylesheet below, for the subtree.
 *
 * The popups of the controls are drawn outside the subtree (a portal on `document.body`); they copy the variables of
 * the enclosing providers onto themselves, so a pinned mode or a mapping reaches them.
 *
 * ## Mapping a host's tokens
 *
 * The figures take their colours from two places, and a host that has its own design tokens connects both, either in
 * its stylesheet as below or with `<ThemeProvider cssVariables={...}>`.
 *
 * Chrome that is HTML (the controls, `FigureFrame`'s title, purpose, readouts and caption, popovers, menus) reads these
 * CSS custom properties, which the page defines (`cssText('dark', '.dark')` from `omai-render/theme` writes a whole
 * set), with the tokens of the current mode as the fallback where one is not defined:
 *
 * - `--wb-fg`: text and icons. `--wb-bg`: the background of fields, menus and switch knobs.
 * - `--wb-surface`: hover and segmented-control backgrounds. `--wb-muted`: labels, units, secondary text.
 * - `--wb-grid`: borders and rules (and a switch's off track). `--wb-accent`: the on state, sliders, focus rings.
 * - `--wb-danger`: a field's validation error. `--wb-font`: the text face of `ThresholdExplorer`'s notes.
 * - `--wb-mono-font`: the face of typeset maths (`maths.css`).
 *
 * `cssVariables` also writes `--wb-fg-secondary`, `--wb-axis`, `--wb-tooltip`, `--wb-success`, `--wb-warning`,
 * `--wb-info` and `--wb-tab0` to `--wb-tab9`, which the package's own styles do not read; they are there for a host's
 * stylesheet. (`ControlPanel` sets `--wb-sidebar` itself, to its sidebar width; do not map it.)
 *
 * Charts and scenes are drawn from the values of `tokens(mode)`, not from CSS variables, so a host's CSS does not
 * recolour them: they follow the mode (`ThemeProvider`, or the page's `dark` class), and a series with an explicit
 * `colour` keeps it. The mode of the figures and the host's must agree: with a host that switches by a class other
 * than `dark`, wrap the figures in `<ThemeProvider mode={...}>`.
 *
 * ```css
 * :root {
 *   --wb-fg: var(--foreground);
 *   --wb-bg: var(--background);
 *   --wb-surface: var(--card);
 *   --wb-muted: var(--muted-foreground);
 *   --wb-grid: var(--border);
 *   --wb-accent: var(--primary);
 *   --wb-danger: var(--destructive);
 * }
 * ```
 */

import {
  type CSSProperties,
  createContext,
  type ReactElement,
  type ReactNode,
  useContext,
  useMemo,
  useSyncExternalStore,
} from 'react'
import { themeName } from '../theme/echarts.ts'
import {
  type ColourMode,
  type CssVariableMap,
  mappedVariables,
  resolveTokenOverride,
  type TokenName,
  type TokenOverride,
  type Tokens,
  tokensWith,
  tokenVariables,
  tokenVariant,
} from '../theme/tokens.ts'

/** The default colour mode when the page says nothing (and during server rendering): dark. */
export const DEFAULT_COLOUR_MODE: ColourMode = 'dark'

function pageMode(): ColourMode {
  if (typeof document === 'undefined') return DEFAULT_COLOUR_MODE
  // The page owns the mode: the `dark` class on the root element means dark, its absence light (the page adds the
  // class at startup, since dark is the site's default).
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light'
}

// One observer of the root element's class for every hook on the page (a page of figures has hundreds of them): made by
// the first subscriber, disconnected by the last.
const pageListeners = new Set<() => void>()
let pageObserver: MutationObserver | undefined

function subscribePage(listener: () => void): () => void {
  if (typeof document === 'undefined' || typeof MutationObserver === 'undefined') return () => {}
  pageListeners.add(listener)
  if (pageObserver === undefined) {
    pageObserver = new MutationObserver(() => {
      for (const l of [...pageListeners]) l()
    })
    pageObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
  }
  return () => {
    pageListeners.delete(listener)
    if (pageListeners.size === 0) {
      pageObserver?.disconnect()
      pageObserver = undefined
    }
  }
}

/**
 * The page's colour mode: the `dark` class on the root element (`<html class="dark">`) means dark and its absence
 * light, observed for changes; dark when there is no page (server rendering). There is no system-preference
 * detection: the page decides.
 *
 * @returns The mode, re-rendering when the page's class changes.
 */
export function usePageColourMode(): ColourMode {
  return useSyncExternalStore(subscribePage, pageMode, () => DEFAULT_COLOUR_MODE)
}

/** What a {@link ThemeProvider} gives the components below it. */
interface ThemeValue {
  readonly mode: ColourMode
  /** The overrides of every provider above, outermost first. */
  readonly overrides: readonly TokenOverride[]
  /** The host variable mappings of every provider above, merged, the inner over the outer. */
  readonly mapping: CssVariableMap
  /** The variables written by every provider above (what a popup outside the subtree must copy). */
  readonly vars: Readonly<Record<string, string>>
}

const ThemeContext = createContext<ThemeValue | undefined>(undefined)

/** Props of {@link ThemeProvider}. */
export interface ThemeProviderProps {
  /** The colour mode below (default the enclosing provider's, else the page's). */
  readonly mode?: ColourMode
  /**
   * A partial set of tokens merged over the mode's: an object, one per mode (`{ light, dark }`) or a function of the
   * mode. Nested providers merge, the inner over the outer.
   */
  readonly tokens?: TokenOverride
  /** Token names mapped to the host's own CSS variables (`{ fg: '--foreground' }`), written as `--wb-fg: var(--foreground)`. */
  readonly cssVariables?: CssVariableMap
  /** Class of the wrapper element. */
  readonly className?: string
  /** Style of the wrapper element (`display: contents` unless it says otherwise). */
  readonly style?: CSSProperties
  readonly children?: ReactNode
}

// A host passes `tokens` and `cssVariables` as literals, new on every render: compare them by content (a function by
// identity) so the context only changes when they do.
function keyOf(value: unknown): unknown {
  return typeof value === 'function' || value === undefined ? value : JSON.stringify(value)
}

/**
 * The theme of the figures and controls below: a colour mode, a restyling of the tokens and a mapping to the host's CSS
 * variables, nestable (see the module comment). Renders one wrapper `div` with `data-theme` and the variables, which is
 * `display: contents` so it does not change the layout.
 *
 * @param props - `mode`, `tokens`, `cssVariables`, the wrapper's `className` and `style`, and the children.
 * @returns The provider.
 * @index function React component: set the colour mode, token overrides and host CSS variables below: `<ThemeProvider mode="light">`
 * @example
 * ```tsx
 * <ThemeProvider mode="dark" tokens={{ accent: '#7c3aed' }} cssVariables={{ fg: '--foreground', muted: '--muted-foreground' }}>
 *   <ThemeProvider mode="light">{slide}</ThemeProvider>
 * </ThemeProvider>
 * ```
 */
export function ThemeProvider(props: ThemeProviderProps): ReactElement {
  const parent = useContext(ThemeContext)
  const page = usePageColourMode()
  const enclosing = parent?.mode ?? page
  const mode = props.mode ?? enclosing
  const tokenKey = keyOf(props.tokens)
  const mapKey = keyOf(props.cssVariables)
  // biome-ignore lint/correctness/useExhaustiveDependencies: `tokenKey` and `mapKey` stand for the props, compared by content
  const value = useMemo((): ThemeValue => {
    const overrides =
      props.tokens === undefined ? (parent?.overrides ?? []) : [...(parent?.overrides ?? []), props.tokens]
    const mapping: CssVariableMap = { ...parent?.mapping, ...props.cssVariables }
    // A provider that moves the mode writes the whole set of variables of the new mode; otherwise only what it overrides.
    const pins = mode !== enclosing
    const own = props.tokens === undefined ? {} : resolveTokenOverride(props.tokens, mode)
    const resolved = tokensWith(mode, ...overrides)
    const written = {
      ...tokenVariables(resolved, pins ? undefined : (Object.keys(own) as TokenName[])),
      ...mappedVariables(pins ? mapping : (props.cssVariables ?? {})),
    }
    return { mode, overrides, mapping, vars: { ...parent?.vars, ...written } }
    // The wrapper's own variables are those of `written`; the context keeps the accumulated set.
  }, [parent, mode, enclosing, tokenKey, mapKey])
  const own = useMemo(() => {
    const out: Record<string, string> = {}
    for (const [k, v] of Object.entries(value.vars)) if (parent?.vars[k] !== v) out[k] = v
    return out
  }, [value, parent])
  return (
    <ThemeContext.Provider value={value}>
      <div
        data-theme={mode}
        className={props.className}
        style={{ display: 'contents', ...own, ...props.style } as CSSProperties}
      >
        {props.children}
      </div>
    </ThemeContext.Provider>
  )
}

/**
 * The colour mode figures should draw in: the nearest {@link ThemeProvider}'s, else the page's (the root element's
 * `dark` class, dark by default).
 *
 * @returns The mode.
 * @index function React hook: the colour mode figures draw in, light or dark: `useColourMode()`
 */
export function useColourMode(): ColourMode {
  const provided = useContext(ThemeContext)
  const page = usePageColourMode()
  return provided?.mode ?? page
}

/**
 * The design tokens of the current colour mode, with the overrides of the enclosing {@link ThemeProvider}s merged in.
 *
 * @param mode - A mode to take them in instead of the current one (a figure with a `mode` prop of its own); the
 *   overrides still apply.
 * @returns The tokens.
 */
export function useTokens(mode?: ColourMode): Tokens {
  const theme = useContext(ThemeContext)
  const page = usePageColourMode()
  const at = mode ?? theme?.mode ?? page
  const overrides = theme?.overrides
  return useMemo(() => tokensWith(at, ...(overrides ?? [])), [at, overrides])
}

/**
 * The CSS variables and `data-theme` that the providers above write on their wrappers, for an element drawn outside
 * their subtree (a popup in a portal on `document.body`) to copy so it looks as it would inside. Empty with no provider.
 *
 * @returns The variables (`--wb-*`), and the mode when there is a provider.
 */
export function useThemeVariables(): { readonly vars: Readonly<Record<string, string>>; readonly mode?: ColourMode } {
  const theme = useContext(ThemeContext)
  return useMemo(() => (theme === undefined ? { vars: {} } : { vars: theme.vars, mode: theme.mode }), [theme])
}

/**
 * The theme a `Chart` registers with ECharts: its tokens (the mode's, with the host's overrides) and the name to
 * register and init them under (`omai-dark`, or `omai-dark-<variant>` when tokens are overridden).
 *
 * @param mode - A mode to take instead of the current one.
 * @returns The mode, the tokens and the theme name.
 */
export function useChartTheme(mode?: ColourMode): {
  readonly mode: ColourMode
  readonly tokens: Tokens
  readonly name: string
} {
  const t = useTokens(mode)
  return useMemo(() => ({ mode: t.mode, tokens: t, name: themeName(t.mode, tokenVariant(t)) }), [t])
}
