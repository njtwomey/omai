/**
 * A table of contents: a plain tree mirroring a directory structure, as given. Folders (nodes with children) open and
 * close; leaves link pages. The gallery gives section folders and recipe files, the docs the engine's directories,
 * modules and symbols; {@link TocSidebar} draws any of them.
 */
/** What a node is, when the tree says: a folder, a file, or one kind of export. Drawn as a glyph before the name. */
export type TocKind = 'folder' | 'file' | 'function' | 'class' | 'interface' | 'type' | 'const' | 'enum'

export interface TocNode {
  /** Unique within the tree. */
  readonly id: string
  readonly title: string
  readonly kind?: TocKind
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

/** Where the current page sits in a tree: its node (if any) and the folders to open to show it. */
export interface TocLocation {
  /** The node that is the current page. */
  readonly id?: string
  /** The folders on the way to it, outermost first, and the node itself when it is a folder. */
  readonly open: readonly string[]
}

/**
 * Find the current page in a tree, whatever kind of node it is: the node linking `path`; else, for a folder page
 * whose folder links nothing (a collection, a directory), the innermost folder holding every page below `path`; else
 * the node linking the longest leading part of `path` (a page inside a page).
 *
 * @param nodes - The tree.
 * @param path - The current page.
 * @returns The node and the folders to open; nothing to open when the page is not in the tree.
 */
export function locate(nodes: readonly TocNode[], path: string): TocLocation {
  const isFolder = (n: TocNode): boolean => n.children !== undefined && n.children.length > 0
  const at = (n: TocNode, trail: readonly string[]): TocLocation => ({
    id: n.id,
    open: isFolder(n) ? [...trail, n.id] : [...trail],
  })
  // The node linking the page itself.
  const exact = (list: readonly TocNode[], trail: readonly string[]): TocLocation | null => {
    for (const n of list) {
      if (n.path === path) return at(n, trail)
      const found = n.children === undefined ? null : exact(n.children, [...trail, n.id])
      if (found !== null) return found
    }
    return null
  }
  const hit = exact(nodes, [])
  if (hit !== null) return hit
  // A folder page: the innermost folder whose pages include every page under `path/`.
  const below = (n: TocNode): string[] => [
    ...(n.path === undefined ? [] : [n.path]),
    ...(n.children ?? []).flatMap(below),
  ]
  const prefix = `${path.replace(/\/$/, '')}/`
  const total = nodes.flatMap(below).filter((p) => p.startsWith(prefix)).length
  if (total > 0) {
    const folder = (list: readonly TocNode[], trail: readonly string[]): TocLocation | null => {
      for (const n of list) {
        if (!isFolder(n)) continue
        const pages = below(n)
        if (pages.filter((p) => p.startsWith(prefix)).length < total) continue
        // This folder holds them all; a child folder may hold them all too.
        return folder(n.children ?? [], [...trail, n.id]) ?? at(n, trail)
      }
      return null
    }
    const found = folder(nodes, [])
    if (found !== null) return found
  }
  // A page within a page: the node linking the longest leading part of the path.
  let best: { loc: TocLocation; length: number } | null = null
  const walk = (list: readonly TocNode[], trail: readonly string[]): void => {
    for (const n of list) {
      if (n.path !== undefined && path.startsWith(`${n.path}/`) && n.path.length > (best?.length ?? 0))
        best = { loc: at(n, trail), length: n.path.length }
      if (n.children !== undefined) walk(n.children, [...trail, n.id])
    }
  }
  walk(nodes, [])
  return (best as { loc: TocLocation } | null)?.loc ?? { open: [] }
}

/**
 * The ids of the folders on the way to the current page (see {@link locate}), outermost first, and the page itself
 * when it is a folder; empty when the page is not in the tree.
 *
 * @param nodes - The tree.
 * @param path - The current page.
 * @returns The ids to open.
 */
export function ancestorsOf(nodes: readonly TocNode[], path: string): string[] {
  return [...locate(nodes, path).open]
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
