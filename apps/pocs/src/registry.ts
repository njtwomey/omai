import { type ComponentType, lazy } from 'react'
import type { Poc, PocEntry } from './poc.ts'

// One folder per proof of concept: `poc.ts` (metadata, loaded now) and `Page.tsx` (the page, loaded when opened).
const metas = import.meta.glob<{ readonly poc: Poc }>('./pocs/*/poc.ts', { eager: true })
const pages = import.meta.glob<{ readonly default: ComponentType }>('./pocs/*/Page.tsx')

const slugOf = (path: string): string => path.split('/').at(-2) as string

/** Every proof of concept, in the order of their folders. */
export const POCS: readonly PocEntry[] = Object.entries(metas).map(([path, m]) => ({ slug: slugOf(path), ...m.poc }))

const components = new Map<string, ComponentType>()

/** The page of a proof of concept (a lazy component, the same one every time), or undefined for an unknown slug. */
export function pageOf(slug: string): ComponentType | undefined {
  const load = pages[`./pocs/${slug}/Page.tsx`]
  if (load === undefined) return undefined
  let c = components.get(slug)
  if (c === undefined) {
    c = lazy(load)
    components.set(slug, c)
  }
  return c
}

/** Slugs that have a metadata file but no page, or the other way round: a mistake to be caught by the test. */
export function mismatched(): string[] {
  const m = new Set(Object.keys(metas).map(slugOf))
  const p = new Set(Object.keys(pages).map(slugOf))
  return [...m].filter((s) => !p.has(s)).concat([...p].filter((s) => !m.has(s)))
}
