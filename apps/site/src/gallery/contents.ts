import type { PageRef } from '../shell/PrevNext.tsx'
import type { TocNode } from '../shell/toc.ts'
import { SECTIONS } from './recipe.ts'
import { entriesOf } from './registry.ts'

/** The gallery's front page, first in reading order. */
export const GALLERY_HOME: PageRef = { title: 'Gallery', path: '/gallery' }

/** The gallery's contents: sections, each with its recipes (for the sidebar and the previous and next links). */
export const GALLERY_TOC: readonly TocNode[] = SECTIONS.filter((s) => entriesOf(s.id).length > 0).map((s) => ({
  id: s.id,
  title: s.title,
  path: `/gallery/${s.id}`,
  keywords: s.blurb,
  children: entriesOf(s.id).map((e) => ({ id: e.path, title: e.title, path: e.path, keywords: e.question })),
}))
