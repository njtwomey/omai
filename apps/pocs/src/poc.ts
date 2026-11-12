/**
 * A proof of concept: one live page showing something the library can do, kept out of the site. The pages live in a tree of
 * directories under `pocs/` (like files in folders): a directory has a `dir.ts` (its title and summary) and holds
 * subdirectories and pocs; a poc is a folder `pocs/<dir>/<subdir>/.../<slug>/` with `poc.ts` (this metadata, small so the
 * index loads all of them) and `Page.tsx` (the page, loaded when opened). The slug is the poc's own folder name, unique across
 * the whole tree, and its address is `/poc/<slug>` wherever it sits. This file is the data model and the pure functions over
 * it (building the tree from the files found, listing a directory, the trail of ancestors, searching); the registry finds the
 * files and the pages draw them.
 */
export interface Poc {
  /** The card's name, e.g. 'Bayesian online changepoint detection'. */
  readonly title: string
  /** One or two sentences: what runs on the page and what to try. */
  readonly summary: string
  /** Search terms beyond the title: the area, the algorithm's aliases, the paper. */
  readonly tags: readonly string[]
  /** The date it was added (YYYY-MM-DD), newest first within its directory. */
  readonly added: string
  /**
   * Legacy: the group this page belonged to before the pages were put in directories. Read only for a page that still sits
   * directly under `pocs/` (not yet moved), which is then listed in the directory of that name; a page in its directory
   * has none.
   */
  readonly group?: string
}

/** What a directory of `pocs/` says about itself, in its `dir.ts` (`export const dir: PocDir`). */
export interface PocDir {
  readonly title: string
  /** One or two sentences: what the directory is about. */
  readonly summary: string
  /** Where it comes among its siblings (ascending); without one, after those that have one, by title. */
  readonly order?: number
}

/** A proof of concept with its address and the directory it is in. */
export interface PocEntry extends Poc {
  readonly slug: string
  /** The directory path, as folder names below `pocs/` (empty: the root). */
  readonly dir: readonly string[]
}

/** The directory of a page that still sits directly under `pocs/` and names no group. */
export const LEGACY_FALLBACK_DIR = 'more'

/** Where a poc file was found: its slug, its directory and whether it is still at the old flat location. */
export interface Location {
  readonly slug: string
  readonly dir: readonly string[]
  /** True when the folder sits directly under `pocs/` (so its directory comes from the legacy `group` field). */
  readonly flat: boolean
}

/**
 * Read where a `poc.ts` or `Page.tsx` lives from its glob key, e.g. `./pocs/expectation-propagation/learned/ep-gaussian-learned/poc.ts`:
 * the slug is its folder, the directory the folders above it. A folder directly under `pocs/` is flat, and gets the directory
 * its legacy group names (none: {@link LEGACY_FALLBACK_DIR}).
 *
 * @param key - The glob key of the file, relative to the app's `src/`.
 * @param group - The legacy `group` field of the poc's metadata, if any.
 * @returns The location.
 */
export function locate(key: string, group?: string): Location {
  const parts = key.replace(/^\.\/pocs\//, '').split('/')
  const folders = parts.slice(0, -1)
  const slug = folders.at(-1) as string
  if (folders.length === 1) return { slug, dir: [group ?? LEGACY_FALLBACK_DIR], flat: true }
  return { slug, dir: folders.slice(0, -1), flat: false }
}

/** The directory a `dir.ts` glob key describes, as a path (`./pocs/a/b/dir.ts` is `['a', 'b']`). */
export function dirOfKey(key: string): string[] {
  return key
    .replace(/^\.\/pocs\//, '')
    .split('/')
    .slice(0, -1)
}

/** A directory of the tree: what it says about itself, its subdirectories and its pocs. */
export interface DirNode {
  /** The folder name (empty for the root). */
  readonly name: string
  /** The path from the root. */
  readonly path: readonly string[]
  readonly meta: PocDir
  /** False when no `dir.ts` describes it (its title is made from its name). */
  readonly declared: boolean
  /** Subdirectories in display order. */
  readonly dirs: readonly DirNode[]
  /** The pocs directly in it, in display order (newest first, then by title). */
  readonly pocs: readonly PocEntry[]
  /** How many pocs there are in it and below it. */
  readonly count: number
}

const byNewest = (a: PocEntry, b: PocEntry): number =>
  a.added === b.added ? a.title.localeCompare(b.title) : a.added < b.added ? 1 : -1

/** 'graphics-and-fields' as 'Graphics and fields': the title of a directory nobody has described. */
const humanise = (name: string): string => {
  const text = name.replace(/[-_]+/g, ' ')
  return text.charAt(0).toUpperCase() + text.slice(1)
}

/** The address of a directory's page (`/` for the root). */
export const dirHref = (path: readonly string[]): string => (path.length === 0 ? '/' : `/dir/${path.join('/')}`)

/** The address of a poc's page. */
export const pocHref = (slug: string): string => `/poc/${slug}`

/**
 * Build the tree of directories. Every directory in `dirs` is in it (empty ones too: a place for pages that are still to come),
 * every directory a poc is in is in it with its ancestors, and one nobody described gets a title made from its name. The
 * directories of a level go by `order` (those without one after, by title); the pocs of a directory newest first, then by title.
 *
 * @param entries - The proofs of concept, each with its directory path.
 * @param dirs - The `dir.ts` metadata, keyed by the directory path joined with `/`.
 * @returns The root.
 */
export function buildTree(entries: readonly PocEntry[], dirs: Readonly<Record<string, PocDir>>): DirNode {
  const known = new Set<string>([''])
  const note = (path: readonly string[]): void => {
    for (let i = 1; i <= path.length; i++) known.add(path.slice(0, i).join('/'))
  }
  for (const key of Object.keys(dirs)) note(key.split('/').filter((s) => s !== ''))
  for (const e of entries) note(e.dir)
  const byOrder = (a: DirNode, b: DirNode): number => {
    const ao = a.meta.order
    const bo = b.meta.order
    if (ao !== undefined && bo !== undefined && ao !== bo) return ao - bo
    if ((ao === undefined) !== (bo === undefined)) return ao === undefined ? 1 : -1
    return a.meta.title.localeCompare(b.meta.title)
  }
  const make = (path: readonly string[]): DirNode => {
    const key = path.join('/')
    const own = dirs[key]
    const children = [...known]
      .filter(
        (k) => k !== '' && k.split('/').length === path.length + 1 && (path.length === 0 || k.startsWith(`${key}/`)),
      )
      .map((k) => make(k.split('/')))
      .sort(byOrder)
    const here = entries.filter((e) => e.dir.join('/') === key).sort(byNewest)
    const name = path.at(-1) ?? ''
    return {
      name,
      path,
      meta: own ?? { title: path.length === 0 ? 'Examples' : humanise(name), summary: '' },
      declared: own !== undefined,
      dirs: children,
      pocs: here,
      count: here.length + children.reduce((n, c) => n + c.count, 0),
    }
  }
  return make([])
}

/** The directory at a path, or undefined when there is none. */
export function findDir(root: DirNode, path: readonly string[]): DirNode | undefined {
  let node: DirNode | undefined = root
  for (const name of path) node = node?.dirs.find((d) => d.name === name)
  return node
}

/** The directories from the root's children down to the one at `path` (the root itself is left out); stops at an unknown name. */
export function ancestors(root: DirNode, path: readonly string[]): DirNode[] {
  const out: DirNode[] = []
  let node: DirNode | undefined = root
  for (const name of path) {
    node = node?.dirs.find((d) => d.name === name)
    if (node === undefined) break
    out.push(node)
  }
  return out
}

/** One step of the trail in the top bar: its label and where it goes (the last one, the page you are on, has no target). */
export interface Crumb {
  readonly label: string
  readonly to: string | undefined
}

/**
 * The trail to a directory or a poc: each ancestor directory (a link to its page), then the poc when one is given (no link:
 * it is the page you are on). The root is the brand of the top bar and not in the trail. A directory page's own crumb has no
 * target either.
 *
 * @param root - The tree.
 * @param path - The directory path (of the poc, when `entry` is given).
 * @param entry - The poc being viewed, if any.
 * @returns The crumbs, outermost first.
 */
export function breadcrumbs(root: DirNode, path: readonly string[], entry?: PocEntry): Crumb[] {
  const dirs = ancestors(root, path)
  const crumbs: Crumb[] = dirs.map((d, i) => ({
    label: d.meta.title,
    to: i === dirs.length - 1 && entry === undefined ? undefined : dirHref(d.path),
  }))
  if (entry !== undefined) crumbs.push({ label: entry.title, to: undefined })
  return crumbs
}

/** The titles of the directories a poc is in, outermost first (the trail without the poc). */
export function dirTitles(root: DirNode, entry: PocEntry): string[] {
  return ancestors(root, entry.dir).map((d) => d.meta.title)
}

/** Every poc in a directory and below it: the directory's own pocs first, then each subdirectory in turn. */
export function pocsBelow(node: DirNode): PocEntry[] {
  return [...node.pocs, ...node.dirs.flatMap(pocsBelow)]
}

/** The slugs that more than one poc has (the address of a poc is its slug, so each must be unique across the whole tree). */
export function duplicateSlugs(entries: readonly PocEntry[]): string[] {
  const seen = new Set<string>()
  const dup = new Set<string>()
  for (const e of entries) (seen.has(e.slug) ? dup : seen).add(e.slug)
  return [...dup]
}

/**
 * The entries that match a search: every word of the query must occur in the title, summary, tags, slug or directory path (case
 * ignored); an empty query keeps all of them. Newest first, then by title. The search is flat: a result carries its directory
 * (`entry.dir`) for the card to show.
 *
 * @param entries - The proofs of concept.
 * @param query - What the user typed.
 * @returns The matching entries in display order.
 */
export function searchPocs(entries: readonly PocEntry[], query: string): PocEntry[] {
  const words = query
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w !== '')
  return entries
    .filter((e) => {
      const hay = `${e.title} ${e.summary} ${e.tags.join(' ')} ${e.slug} ${e.dir.join(' ')}`.toLowerCase()
      return words.every((w) => hay.includes(w))
    })
    .sort(byNewest)
}
