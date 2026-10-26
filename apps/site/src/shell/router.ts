import { useSyncExternalStore } from 'react'

const listeners = new Set<() => void>()

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  window.addEventListener('popstate', listener)
  return () => {
    listeners.delete(listener)
    window.removeEventListener('popstate', listener)
  }
}

const base = import.meta.env.BASE_URL.replace(/\/$/, '')

/** The current path within the site (without the deploy base), e.g. `/gallery/charts`, outside React. */
export function currentPath(): string {
  return window.location.pathname.slice(base.length) || '/'
}

/** The current path within the site (without the deploy base), e.g. `/gallery/charts`. */
export function usePath(): string {
  return useSyncExternalStore(subscribe, currentPath)
}

/** Go to a path within the site without reloading. */
export function navigate(path: string): void {
  window.history.pushState(null, '', `${base}${path}`)
  for (const l of listeners) l()
  window.scrollTo(0, 0)
}

/** A site-relative href with the deploy base, for anchors. */
export function href(path: string): string {
  return `${base}${path}`
}
