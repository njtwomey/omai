import { describe, expect, it } from 'vitest'
import { SECTIONS } from './sections.ts'
import { bySection, neighbours, readingOrder, type StoryEntry, storyHref, storySlugOf } from './story.ts'

const entry = (slug: string, section: StoryEntry['section'], date: string): StoryEntry => ({
  slug,
  title: slug.toUpperCase(),
  standfirst: `about ${slug}`,
  section,
  tags: [],
  date,
})

const stories = [
  entry('c', 'inference', '2026-11-01'),
  entry('a', 'uncertainty', '2026-11-02'),
  entry('b', 'uncertainty', '2026-11-05'),
  entry('d', 'inference', '2026-10-01'),
]

describe('the story listing', () => {
  it('groups by section in the order of sections.ts, newest first, leaving out empty sections', () => {
    const groups = bySection(stories)
    expect(groups.map((g) => g.id)).toEqual(['uncertainty', 'inference'])
    expect(groups.map((g) => g.stories.map((s) => s.slug))).toEqual([
      ['b', 'a'],
      ['c', 'd'],
    ])
    const order = SECTIONS.map((s) => s.id)
    expect(groups.map((g) => g.id)).toEqual(order.filter((id) => groups.some((g) => g.id === id)))
  })

  it('reads in the order of the splash and finds the neighbours of a story', () => {
    expect(readingOrder(stories).map((s) => s.slug)).toEqual(['b', 'a', 'c', 'd'])
    expect(neighbours(stories, 'b')).toMatchObject({ prev: undefined, next: { slug: 'a' } })
    expect(neighbours(stories, 'a')).toMatchObject({ prev: { slug: 'b' }, next: { slug: 'c' } })
    expect(neighbours(stories, 'd')).toMatchObject({ prev: { slug: 'c' }, next: undefined })
    expect(neighbours(stories, 'nope')).toEqual({ prev: undefined, next: undefined })
  })

  it('knows the routes', () => {
    expect(storyHref('a')).toBe('/story/a')
    expect(storySlugOf('/story/a')).toBe('a')
    expect(storySlugOf('/story/a/')).toBe('a')
    expect(storySlugOf('/')).toBeUndefined()
    expect(storySlugOf('/story/a/b')).toBeUndefined()
  })
})
