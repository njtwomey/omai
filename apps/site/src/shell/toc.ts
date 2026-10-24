/**
 * A table of contents: a tree of titled nodes, the leaves (and any inner node with a page) linking a path. The
 * gallery gives sections and recipes, the hub collections, sub-collections and entries, the docs packages, domains,
 * modules and symbols; {@link TocSidebar} draws any of them.
 */
export interface TocNode {
  /** Unique within the tree (remembered open/closed state is keyed by it). */
  readonly id: string
  readonly title: string
  /** The page the node links, if any. */
  readonly path?: string
  /** A short note after the title (a count, a kind). */
  readonly badge?: string
  /** Extra text the search matches besides the title (a question, tags). */
  readonly keywords?: string
  readonly children?: readonly TocNode[]
}

/**
 * The tree cut down to the nodes matching a query (every word of it, case-insensitive, in the title or keywords) and
 * their ancestors; a matching inner node keeps all its children. An empty query keeps everything.
 *
 * @param nodes - The tree.
 * @param query - What was typed.
 * @returns The filtered tree.
 */
export function filterToc(nodes: readonly TocNode[], query: string): TocNode[] {
  const words = query
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w !== '')
  if (words.length === 0) return [...nodes]
  const matches = (n: TocNode): boolean => {
    const text = `${n.title} ${n.keywords ?? ''}`.toLowerCase()
    return words.every((w) => text.includes(w))
  }
  const walk = (list: readonly TocNode[]): TocNode[] =>
    list.flatMap((n): TocNode[] => {
      if (matches(n)) return [n]
      const children = n.children === undefined ? [] : walk(n.children)
      return children.length > 0 ? [{ ...n, children }] : []
    })
  return walk(nodes)
}

/**
 * The ids of the nodes on the way to the node linking `path` (the node itself excluded), outermost first; empty when
 * no node links it.
 *
 * @param nodes - The tree.
 * @param path - The current page.
 * @returns The ancestors' ids.
 */
export function ancestorsOf(nodes: readonly TocNode[], path: string): string[] {
  const walk = (list: readonly TocNode[], trail: readonly string[]): string[] | null => {
    for (const n of list) {
      if (n.path === path) return [...trail]
      if (n.children !== undefined) {
        const found = walk(n.children, [...trail, n.id])
        if (found !== null) return found
      }
    }
    return null
  }
  return walk(nodes, []) ?? []
}

/**
 * Every inner node's id (to open the whole of a filtered tree).
 *
 * @param nodes - The tree.
 * @returns The ids of nodes with children.
 */
export function innerIds(nodes: readonly TocNode[]): string[] {
  return nodes.flatMap((n) =>
    n.children === undefined || n.children.length === 0 ? [] : [n.id, ...innerIds(n.children)],
  )
}
