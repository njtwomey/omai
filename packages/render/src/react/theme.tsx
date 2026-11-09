/**
 * The colour mode in React: `ThemeProvider` and `useColourMode` take the mode from a prop or from the page's root
 * `dark` class (dark by default, no system-preference detection), and `useTokens` gives the design tokens of the
 * current mode.
 *
 * ## Mapping a host's tokens
 *
 * The figures take their colours from two places, and a host that has its own design tokens connects both.
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

import { createContext, type ReactElement, type ReactNode, useContext, useMemo, useSyncExternalStore } from 'react'
import { type ColourMode, type Tokens, tokens } from '../theme/tokens.ts'

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

const ColourModeContext = createContext<ColourMode | undefined>(undefined)

/**
 * Supply a colour mode to the figures below: an explicit `mode`, or (without one) the page's mode from the root
 * element's `dark` class.
 *
 * @param props - `mode` to force a mode, and the children.
 * @returns The provider.
 * @index function React component: force a colour mode for the figures inside: `<ThemeProvider mode="light">...</ThemeProvider>`
 */
export function ThemeProvider(props: { readonly mode?: ColourMode; readonly children?: ReactNode }): ReactElement {
  const page = usePageColourMode()
  return <ColourModeContext.Provider value={props.mode ?? page}>{props.children}</ColourModeContext.Provider>
}

/**
 * The colour mode figures should draw in: the nearest {@link ThemeProvider}'s, else the page's (the root element's
 * `dark` class, dark by default).
 *
 * @returns The mode.
 * @index function React hook: the colour mode figures draw in, light or dark: `useColourMode()`
 */
export function useColourMode(): ColourMode {
  const provided = useContext(ColourModeContext)
  const page = usePageColourMode()
  return provided ?? page
}

/**
 * The design tokens of the current colour mode.
 *
 * @returns The tokens.
 */
export function useTokens(): Tokens {
  const mode = useColourMode()
  return useMemo(() => tokens(mode), [mode])
}
