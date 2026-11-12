/**
 * The stories of the app, found from the folders of `stories/`: each holds `story.ts` (its metadata, loaded now, so the splash
 * can list it) and `Story.tsx` (the narrative, loaded when the story is opened). The `compute.ts` and `worker.ts` of a story are
 * never imported here: the worker's bundle, with the engine in it, is fetched only when the reader presses play.
 */

import { type ComponentType, lazy } from 'react'
import type { StoryEntry, StoryMeta } from './story.ts'

const metas = import.meta.glob<{ readonly story: StoryMeta }>('./stories/*/story.ts', { eager: true })
const pages = import.meta.glob<{ readonly default: ComponentType }>('./stories/*/Story.tsx')

const slugOf = (key: string): string => key.split('/').at(-2) as string

/** Every story, in the order of their folders. */
export const STORIES: readonly StoryEntry[] = Object.entries(metas).map(([key, m]) => ({
  ...m.story,
  slug: slugOf(key),
}))

const loaders = new Map(Object.entries(pages).map(([key, load]) => [slugOf(key), load]))
const components = new Map<string, ComponentType>()

/** The story of a slug, or undefined. */
export const storyOf = (slug: string): StoryEntry | undefined => STORIES.find((s) => s.slug === slug)

/** The page of a story (a lazy component, the same one every time), or undefined for an unknown slug. */
export function pageOf(slug: string): ComponentType | undefined {
  const load = loaders.get(slug)
  if (load === undefined) return undefined
  let c = components.get(slug)
  if (c === undefined) {
    c = lazy(load)
    components.set(slug, c)
  }
  return c
}

/** Folders that have a metadata file but no narrative, or the other way round: a mistake the registry test catches. */
export function mismatched(): string[] {
  const m = new Set(Object.keys(metas).map(slugOf))
  const p = new Set(Object.keys(pages).map(slugOf))
  return [...m].filter((s) => !p.has(s)).concat([...p].filter((s) => !m.has(s)))
}
