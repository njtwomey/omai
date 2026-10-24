/**
 * The docs table of contents as data: package, then the core's domains by tier and the methods' areas, then each
 * module's subfolders, then its symbols. The shared sidebar component renders it.
 */
import type { ModuleDoc } from 'whitebox-meta/schema'
import { type Docs, modulePath, symbolPath } from './data.ts'

/** A node of the table of contents. */
export interface TocNode {
  readonly id: string
  readonly title: string
  readonly path?: string
  readonly children?: readonly TocNode[]
}

function moduleNode(docs: Docs, m: ModuleDoc, withSymbols: boolean): TocNode {
  const children: TocNode[] = m.children.flatMap((c) => {
    const child = docs.module(c)
    return child === undefined ? [] : [moduleNode(docs, child, withSymbols)]
  })
  if (withSymbols) for (const s of docs.symbolsOf(m.id)) children.push({ id: s.id, title: s.name, path: symbolPath(s) })
  return { id: m.id, title: m.name, path: modulePath(m.id), ...(children.length > 0 ? { children } : {}) }
}

/**
 * The tree: Core (tier groups of domains) and Methods (areas), each module with its subfolders and symbols.
 *
 * @param docs - The docs index.
 * @param withSymbols - Whether to list symbols under their modules (default true).
 * @returns The top-level nodes.
 */
export function docsToc(docs: Docs, withSymbols = true): TocNode[] {
  const top = docs.modules.filter((m) => m.parent === undefined)
  const core = top.filter((m) => m.layer === 'core')
  const tiers = [...new Set(core.map((m) => m.tier ?? 99))].sort((a, b) => a - b)
  return [
    {
      id: 'core',
      title: 'Core',
      children: tiers.map((t) => ({
        id: `tier-${t}`,
        title: `Tier ${t}`,
        children: core.filter((m) => (m.tier ?? 99) === t).map((m) => moduleNode(docs, m, withSymbols)),
      })),
    },
    {
      id: 'methods',
      title: 'Methods',
      children: top.filter((m) => m.layer === 'methods').map((m) => moduleNode(docs, m, withSymbols)),
    },
  ]
}
