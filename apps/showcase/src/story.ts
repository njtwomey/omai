/**
 * What a story is, and the pure functions that list stories: the metadata a story's `story.ts` exports, grouping by section in
 * the order of `sections.ts`, and the neighbours of a story in reading order. Nothing here touches the page or the files, so
 * `registry.ts` (which finds the stories) and the splash and story pages share it and `story.test.ts` can test it on fixtures.
 */

import { SECTIONS, type SectionId } from './sections.ts'

/** The metadata of a story: what the splash card and the head of the page say. */
export interface StoryMeta {
  /** The title of the post, a few words. */
  readonly title: string
  /** One line under the title; also the hook on the story's card unless `hook` is given. */
  readonly standfirst: string
  /** The section of the splash page it is listed in (an id of `sections.ts`). */
  readonly section: SectionId
  /** Words a reader might search for. */
  readonly tags: readonly string[]
  /** When it was written, `YYYY-MM-DD`; the newest story of a section comes first. */
  readonly date: string
  /** The line shown after the title in the list (the standfirst when there is none). */
  readonly hook?: string
  /** A story that opens its section: it is listed before the others, whatever their dates. */
  readonly first?: boolean
}

/** A story as the app knows it: its metadata and its slug (the name of its folder). */
export interface StoryEntry extends StoryMeta {
  readonly slug: string
}

/** The address of a story. */
export const storyHref = (slug: string): string => `/story/${slug}`

/** The slug of `/story/<slug>`, or undefined for any other path. */
export function storySlugOf(path: string): string | undefined {
  const m = /^\/story\/([^/]+)\/?$/.exec(path)
  return m === null ? undefined : (m[1] as string)
}

/** One section of the splash page with its stories. */
export interface SectionGroup {
  readonly id: SectionId
  readonly title: string
  readonly blurb: string
  readonly stories: readonly StoryEntry[]
}

/**
 * The stories grouped by section, in the section order of `sections.ts`, newest first within a section; a section with no
 * story is left out.
 *
 * @param stories - Every story.
 * @returns The groups.
 */
export function bySection(stories: readonly StoryEntry[]): SectionGroup[] {
  return SECTIONS.map((s) => ({
    ...s,
    stories: stories
      .filter((e) => e.section === s.id)
      .sort(
        (a, b) =>
          Number(b.first === true) - Number(a.first === true) ||
          b.date.localeCompare(a.date) ||
          a.title.localeCompare(b.title),
      ),
  })).filter((g) => g.stories.length > 0)
}

/**
 * The stories in reading order: the order the splash shows them (section by section).
 *
 * @param stories - Every story.
 * @returns The same stories, flat.
 */
export const readingOrder = (stories: readonly StoryEntry[]): StoryEntry[] =>
  bySection(stories).flatMap((g) => g.stories)

/**
 * The story before and after one in reading order (undefined at the ends).
 *
 * @param stories - Every story.
 * @param slug - The slug of the story.
 * @returns The neighbours.
 */
export function neighbours(
  stories: readonly StoryEntry[],
  slug: string,
): { readonly prev: StoryEntry | undefined; readonly next: StoryEntry | undefined } {
  const order = readingOrder(stories)
  const at = order.findIndex((e) => e.slug === slug)
  return { prev: at > 0 ? order[at - 1] : undefined, next: at >= 0 ? order[at + 1] : undefined }
}
