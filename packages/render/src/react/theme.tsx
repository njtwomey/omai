/**
 * The colour mode in React: `ThemeProvider` and `useColourMode` take the mode from a prop or from the page's root
 * `dark` class (dark by default, no system-preference detection), and `useTokens` gives the design tokens of the
 * current mode.
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
