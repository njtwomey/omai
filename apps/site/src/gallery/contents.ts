import type { PageRef } from '../shell/PrevNext.tsx'
import type { TocNode } from '../shell/toc.ts'
import { entriesOf, SECTIONS } from './registry.ts'

/** The gallery's front page, first in reading order. */
export const GALLERY_HOME: PageRef = { title: 'Gallery', path: '/gallery' }

/** The gallery's contents as its folders: each section folder with its recipe files, in on-disk order. */
export const GALLERY_TOC: readonly TocNode[] = SECTIONS.filter((s) => entriesOf(s.id).length > 0).map((s) => ({
  id: s.id,
  title: s.id,
  children: entriesOf(s.id).map((e) => ({ id: e.path, title: e.title, path: e.path })),
}))
