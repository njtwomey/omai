import type { ReactNode } from 'react'
import { TocSidebar } from '../shell/TocSidebar.tsx'
import type { TocNode } from '../shell/toc.ts'
import { EntryPage, GalleryHome, SectionPage } from './pages.tsx'
import { SECTIONS } from './recipe.ts'
import { ENTRIES, entriesOf } from './registry.ts'

/** The gallery section: renders the page for a path under /gallery. */
export function GallerySection(props: { readonly path: string }): ReactNode {
  const parts = props.path
    .replace(/^\/gallery\/?/, '')
    .replace(/\/$/, '')
    .split('/')
    .filter((p) => p !== '')
  const entry = ENTRIES.find((e) => e.path === `/gallery/${parts.join('/')}`)
  if (entry !== undefined) return <EntryPage entry={entry} />
  const section = SECTIONS.find((s) => s.id === parts[0])
  if (section !== undefined) return <SectionPage section={section} />
  return <GalleryHome />
}

/** The gallery's contents: sections, each with its recipes. */
export const GALLERY_TOC: readonly TocNode[] = SECTIONS.filter((s) => entriesOf(s.id).length > 0).map((s) => ({
  id: s.id,
  title: s.title,
  path: `/gallery/${s.id}`,
  keywords: s.blurb,
  children: entriesOf(s.id).map((e) => ({ id: e.path, title: e.title, path: e.path, keywords: e.question })),
}))

/** The gallery's sidebar. */
export function GalleryToc(props: { readonly path: string }): ReactNode {
  return (
    <TocSidebar
      nodes={GALLERY_TOC}
      current={props.path.replace(/\/$/, '')}
      storageKey="gallery"
      home={{ title: 'Gallery', path: '/gallery' }}
      placeholder="Search recipes"
    />
  )
}
