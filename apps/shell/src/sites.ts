/** The three whitebox-js sites, in the order the navigation shows them. */
export type SiteId = 'gallery' | 'hub' | 'docs'

/** A site in the shared navigation. */
export interface Site {
  readonly id: SiteId
  readonly title: string
  readonly url: string
}

const env = (import.meta as { env?: Record<string, string | undefined> }).env ?? {}

/** The sites, with URLs from `VITE_WHITEBOX_<ID>_URL` (defaults: the local dev servers). */
export const sites: readonly Site[] = [
  { id: 'gallery', title: 'Gallery', url: env['VITE_WHITEBOX_GALLERY_URL'] ?? 'http://localhost:5301/' },
  { id: 'hub', title: 'Hub', url: env['VITE_WHITEBOX_HUB_URL'] ?? 'http://localhost:5302/' },
  { id: 'docs', title: 'Docs', url: env['VITE_WHITEBOX_DOCS_URL'] ?? 'http://localhost:5303/' },
]
