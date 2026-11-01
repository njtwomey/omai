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
  /** The date it was added (YYYY-MM-DD), newest first on the index. */
  readonly added: string
}

/** A proof of concept with its address. */
export interface PocEntry extends Poc {
  readonly slug: string
}

/**
 * The entries that match a search: every word of the query must occur in the title, summary, tags or slug (case
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
      const hay = `${e.title} ${e.summary} ${e.tags.join(' ')} ${e.slug}`.toLowerCase()
      return words.every((w) => hay.includes(w))
    })
    .sort((a, b) => (a.added === b.added ? a.title.localeCompare(b.title) : a.added < b.added ? 1 : -1))
}
