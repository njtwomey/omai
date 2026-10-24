/**
 * A table of contents: a plain tree mirroring a directory structure, as given. Folders (nodes with children) open and
 * close; leaves link pages. The gallery gives section folders and recipe files, the docs the engine's directories,
 * modules and symbols; {@link TocSidebar} draws any of them.
 */
export interface TocNode {
  /** Unique within the tree. */
  readonly id: string
  readonly title: string
  /** The page the node links, if any (usually the leaves). */
  readonly path?: string
  readonly children?: readonly TocNode[]
}

/**
 * The tree cut down to the nodes whose title contains the query (case-insensitive) and their ancestors; a matching
 * folder keeps all its children. An empty query keeps everything.
 *
 * @param nodes - The tree.
 * @param query - What was typed.
 * @returns The filtered tree.
 */
export function filterToc(nodes: readonly TocNode[], query: string): TocNode[] {
  const q = query.trim().toLowerCase()
  if (q === '') return [...nodes]
  const walk = (list: readonly TocNode[]): TocNode[] =>
    list.flatMap((n): TocNode[] => {
      if (n.title.toLowerCase().includes(q)) return [n]
      const children = n.children === undefined ? [] : walk(n.children)
      return children.length > 0 ? [{ ...n, children }] : []
    })
  return walk(nodes)
}

/**
 * The ids of the folders on the way to the node linking `path`, outermost first, and the node itself when it is a
 * folder; empty when no node links it.
 *
 * @param nodes - The tree.
 * @param path - The current page.
 * @returns The ids to open.
 */
export function ancestorsOf(nodes: readonly TocNode[], path: string): string[] {
  const walk = (list: readonly TocNode[], trail: readonly string[]): string[] | null => {
    for (const n of list) {
      if (n.path === path) return n.children !== undefined && n.children.length > 0 ? [...trail, n.id] : [...trail]
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
 * Every folder's id (to open the whole of a filtered tree).
 *
 * @param nodes - The tree.
 * @returns The ids of nodes with children.
 */
export function innerIds(nodes: readonly TocNode[]): string[] {
  return nodes.flatMap((n) =>
    n.children === undefined || n.children.length === 0 ? [] : [n.id, ...innerIds(n.children)],
  )
}
