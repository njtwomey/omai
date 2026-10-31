import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { viewedKinds, viewFor } from 'mdml-render/views'
import { describe, expect, it } from 'vitest'
import { ENTRIES } from './registry.ts'

/**
 * Everything the engine can explain, the gallery must show: every exposition record kind (and so its standard view) is
 * named by some recipe's `shows`, and so is every other view component. A recipe lists what it really draws; this test
 * fails when a new kind or view is added without a page.
 */

/** Record kinds allowed to have no recipe, each with the reason. Empty unless a kind truly cannot be shown. */
const KIND_EXEMPTIONS: Readonly<Record<string, string>> = {}

/** View components allowed to have no recipe of their own, each with the reason. */
const VIEW_EXEMPTIONS: Readonly<Record<string, string>> = {
  FramesPlayer:
    'the building block under every playing view (ClusteringPlayer, BoardPlayer and the rest), not a view of any record of its own; each of those has its page',
}

const ROOT = join(import.meta.dirname, '..', '..', '..', '..')
const read = (dir: string): string[] =>
  readdirSync(join(ROOT, dir))
    .filter((f) => /\.tsx?$/.test(f) && !/\.test(-d)?\.tsx?$/.test(f))
    .map((f) => readFileSync(join(ROOT, dir, f), 'utf8'))

const shown = new Set<string>(ENTRIES.flatMap((e) => e.shows ?? []))

describe('the gallery shows every record kind and view', () => {
  it('derives the record kinds from the engine, not from a list kept here', () => {
    // The kinds named by the exposition interfaces (each begins with `readonly kind: '<kind>'`), read from the source,
    // against the kinds the render registry has a view for (exhaustive over the `Exposition` union by its type).
    const declared = new Set<string>()
    for (const text of read('packages/engine/src/lib/exposition'))
      for (const m of text.matchAll(/export interface \w+ \{\s*(?:\/\*\*[\s\S]*?\*\/\s*)?readonly kind: '(\w+)'/g))
        declared.add(m[1] as string)
    expect(declared.size).toBeGreaterThan(50)
    expect([...declared].sort()).toEqual([...viewedKinds()].sort())
  })

  it('names every record kind in some recipe', () => {
    const missing = viewedKinds().filter((k) => !shown.has(k) && KIND_EXEMPTIONS[k] === undefined)
    expect(missing, `record kinds no recipe shows: add them to a recipe's \`shows\`, or a page`).toEqual([])
  })

  it('names every view component, the standard one of a shown kind or a recipe of its own', () => {
    const standard = new Set(
      viewedKinds()
        .filter((k) => shown.has(k))
        .map((k) => viewFor(k).component),
    )
    const views = new Set<string>()
    for (const text of read('packages/render/src/react/views'))
      for (const m of text.matchAll(/^export function (\w+(?:View|Player))\(/gm)) views.add(m[1] as string)
    expect(views.size).toBeGreaterThan(40)
    const missing = [...views].filter((v) => !shown.has(v) && !standard.has(v) && VIEW_EXEMPTIONS[v] === undefined)
    expect(missing, 'views no recipe shows').toEqual([])
  })

  it('lists no exemption that a recipe already covers, and no name that is not a kind or view', () => {
    const kinds = new Set<string>(viewedKinds())
    for (const k of Object.keys(KIND_EXEMPTIONS)) {
      expect(kinds.has(k), `${k} is not a record kind`).toBe(true)
      expect(shown.has(k), `${k} is exempt but a recipe shows it`).toBe(false)
    }
    for (const v of Object.keys(VIEW_EXEMPTIONS))
      expect(shown.has(v), `${v} is exempt but a recipe shows it`).toBe(false)
  })

  it('only lists record kinds and views that exist in the recipes', () => {
    const kinds = new Set<string>(viewedKinds())
    const components = new Set<string>(viewedKinds().map((k) => viewFor(k).component))
    const allViews = new Set<string>()
    for (const text of read('packages/render/src/react'))
      for (const m of text.matchAll(/^export function (\w+)\(/gm)) allViews.add(m[1] as string)
    for (const text of read('packages/render/src/react/views'))
      for (const m of text.matchAll(/^export function (\w+)\(/gm)) allViews.add(m[1] as string)
    for (const e of ENTRIES)
      for (const name of e.shows ?? [])
        expect(kinds.has(name) || components.has(name) || allViews.has(name), `${e.file} shows unknown ${name}`).toBe(
          true,
        )
  })
})
