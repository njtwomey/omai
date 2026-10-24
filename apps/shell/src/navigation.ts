import { type MouseEvent, useEffect, useSyncExternalStore } from 'react'

/**
 * In-app paths: each site is a single-page app whose pages are paths under Vite's `base` (`/` in development, a
 * sub-path when deployed). A path here never has slashes at either end: `''` is the site's front page.
 */

const env = (import.meta as { env?: Record<string, string | undefined> }).env ?? {}
const BASE = env['BASE_URL'] ?? '/'

/** The URL of an in-app path, with an optional `#hash`. */
export function hrefOf(path: string, hash = ''): string {
  const clean = path.replace(/^\/+|\/+$/g, '')
  return `${BASE}${clean}${hash === '' ? '' : `#${hash}`}`
}

/** The current in-app path. */
export function currentPath(): string {
  if (typeof location === 'undefined') return ''
  const { pathname } = location
  const local = pathname.startsWith(BASE) ? pathname.slice(BASE.length) : pathname
  return decodeURIComponent(local.replace(/^\/+|\/+$/g, ''))
}

/** Open an in-app path without reloading (the shell listens for `popstate`). */
export function go(path: string, hash = ''): void {
  const url = hrefOf(path, hash)
  if (url !== `${location.pathname}${location.hash}`) history.pushState(null, '', url)
  dispatchEvent(new PopStateEvent('popstate'))
}

/** A click handler that follows a plain click on an in-app link in place (modified clicks open new tabs as usual). */
export function onLink(path: string, hash = ''): (e: MouseEvent) => void {
  return (e) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return
    e.preventDefault()
    go(path, hash)
  }
}

function subscribe(onChange: () => void): () => void {
  addEventListener('popstate', onChange)
  return () => removeEventListener('popstate', onChange)
}

/**
 * The current in-app path, re-rendering on navigation. A change of path scrolls to the top, or to the element named
 * by the URL's hash once the new page has drawn.
 */
export function usePath(): string {
  const path = useSyncExternalStore(subscribe, currentPath, () => '')
  // biome-ignore lint/correctness/useExhaustiveDependencies: runs on each change of path, which the effect reads from `location`
  useEffect(() => {
    const hash = location.hash.slice(1)
    const frame = requestAnimationFrame(() => {
      const target = hash === '' ? null : document.getElementById(decodeURIComponent(hash))
      if (target === null) scrollTo({ top: 0 })
      else target.scrollIntoView({ block: 'start' })
    })
    return () => cancelAnimationFrame(frame)
  }, [path])
  return path
}
