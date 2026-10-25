import { type ReactNode, useEffect } from 'react'
import { PrevNext } from '../shell/PrevNext.tsx'
import { navigate } from '../shell/router.ts'
import { TocSidebar } from '../shell/TocSidebar.tsx'
import { GALLERY_HOME, GALLERY_TOC } from './contents.ts'
import { MOVED } from './moved.ts'
import { EntryPage, GalleryHome, SectionPage } from './pages.tsx'
import { ENTRIES, SECTIONS } from './registry.ts'

/** The gallery section: renders the page for a path under /gallery, with previous and next links. */
export function GallerySection(props: { readonly path: string }): ReactNode {
  const path = props.path.replace(/\/$/, '')
  const moved = MOVED[path]
  useEffect(() => {
    if (moved !== undefined) navigate(moved)
  }, [moved])
  const parts = path
    .replace(/^\/gallery\/?/, '')
    .split('/')
    .filter((p) => p !== '')
  const nav = <PrevNext nodes={GALLERY_TOC} home={GALLERY_HOME} current={path} compact />
  const entry = ENTRIES.find((e) => e.path === path)
  const section = SECTIONS.find((s) => s.id === parts[0])
  const page =
    entry !== undefined ? (
      <EntryPage entry={entry} nav={nav} />
    ) : section !== undefined ? (
      <SectionPage section={section} nav={nav} />
    ) : (
      <GalleryHome nav={nav} />
    )
  return (
    <>
      {page}
      <PrevNext nodes={GALLERY_TOC} home={GALLERY_HOME} current={(entry ?? section) ? path : GALLERY_HOME.path} />
    </>
  )
}

/** The gallery's sidebar. */
export function GalleryToc(props: { readonly path: string }): ReactNode {
  return (
    <TocSidebar
      nodes={GALLERY_TOC}
      current={props.path.replace(/\/$/, '')}
      home={GALLERY_HOME}
      placeholder="Filter recipes"
    />
  )
}
