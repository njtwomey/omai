/**
 * A proof of concept: one live page showing something the library can do, kept out of the site. Each lives in
 * `pocs/<slug>/` as `poc.ts` (this metadata, small so the index loads all of them) and `Page.tsx` (the page, loaded
 * when opened). The index lists them and searches the title, summary and tags.
 */
export interface Poc {
  /** The card's name, e.g. 'Bayesian online changepoint detection'. */
  readonly title: string
  /** One or two sentences: what runs on the page and what to try. */
  readonly summary: string
  /** Search terms beyond the title: the area, the algorithm's aliases, the paper. */
  readonly tags: readonly string[]
  /** The date it was added (YYYY-MM-DD), newest first within its group. */
  readonly added: string
  /** The slug of its group in `groups.ts` (the index lists a group's pocs together); none: listed under "More". */
  readonly group?: string
}

/** A group of proofs of concept that belong together, shown as one section of the index (and at `/group/<slug>`). */
export interface PocGroup {
  readonly slug: string
  readonly title: string
  /** One or two sentences: what the group is about. */
  readonly summary: string
}

/** One section of the index: a group (or the ungrouped rest) with its matching pocs in display order. */
export interface PocSection {
  /** The group, or undefined for the proofs of concept that belong to none. */
  readonly group: PocGroup | undefined
  readonly entries: readonly PocEntry[]
}

/**
 * Split matching proofs of concept into sections: one per declared group, in the order of `groups`, then the ones that
 * belong to no group (or to an undeclared one). A section with no entries is left out. Within a section the order of
 * `entries` is kept (hand it the result of {@link searchPocs}).
 *
 * @param entries - The proofs of concept, in display order.
 * @param groups - The declared groups, in display order.
 * @returns The non-empty sections.
 */
export function groupPocs(entries: readonly PocEntry[], groups: readonly PocGroup[]): PocSection[] {
  const known = new Set(groups.map((g) => g.slug))
  const sections: PocSection[] = groups.map((g) => ({ group: g, entries: entries.filter((e) => e.group === g.slug) }))
  sections.push({ group: undefined, entries: entries.filter((e) => e.group === undefined || !known.has(e.group)) })
  return sections.filter((s) => s.entries.length > 0)
}

/** A proof of concept with its address. */
export interface PocEntry extends Poc {
  readonly slug: string
}

/**
 * The entries that match a search: every word of the query must occur in the title, summary, tags, slug or group (case
 * ignored); an empty query keeps all of them. Newest first, then by title.
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
      const hay = `${e.title} ${e.summary} ${e.tags.join(' ')} ${e.slug} ${e.group ?? ''}`.toLowerCase()
      return words.every((w) => hay.includes(w))
    })
    .sort((a, b) => (a.added === b.added ? a.title.localeCompare(b.title) : a.added < b.added ? 1 : -1))
}
