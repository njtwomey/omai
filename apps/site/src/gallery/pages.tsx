import type { ReactNode } from 'react'
import { Thumbnail } from '../home/Thumbnail.tsx'
import { LazyRecipeView } from '../shell/LazyRecipe.tsx'
import { Card, CardGrid, PageHeader, PageSection, Prose } from '../shell/layout.tsx'
import type { Section } from './recipe.ts'
import { ENTRIES, type Entry, entriesOf, SECTIONS } from './registry.ts'

/** The gallery's front page: each section with its recipes as cards. */
export function GalleryHome(props: { readonly nav?: ReactNode }): ReactNode {
  return (
    <>
      <PageHeader
        title="The render gallery"
        actions={props.nav}
        lede={`${ENTRIES.length} small live recipes for omai-render: what it draws, how figures are laid out, and how its controls, interactions and players work. Each page shows the figure beside the code that makes it.`}
      />
      {SECTIONS.map((s) => {
        const items = entriesOf(s.id)
        if (items.length === 0) return null
        return (
          <PageSection key={s.id} id={s.id} title={s.title} blurb={s.blurb}>
            <CardGrid min={260}>
              {items.map((e) => (
                <Card
                  key={e.path}
                  path={e.path}
                  title={e.title}
                  text={e.question}
                  preview={<Thumbnail section={e.section} slug={e.slug} />}
                />
              ))}
            </CardGrid>
          </PageSection>
        )
      })}
    </>
  )
}

/** One section: its recipes. */
export function SectionPage(props: { readonly section: Section; readonly nav?: ReactNode }): ReactNode {
  const { section } = props
  const items = entriesOf(section.id)
  return (
    <>
      <PageHeader
        title={section.title}
        lede={section.blurb}
        eyebrow={{ label: 'Gallery', path: '/gallery' }}
        actions={props.nav}
      />
      {items.length === 0 ? (
        <p className="text-sm text-muted">No recipes here yet.</p>
      ) : (
        <CardGrid min={260}>
          {items.map((e) => (
            <Card
              key={e.path}
              path={e.path}
              title={e.title}
              text={e.question}
              preview={<Thumbnail section={e.section} slug={e.slug} />}
            />
          ))}
        </CardGrid>
      )}
    </>
  )
}

/** One recipe: the question, a sentence or two, the live example beside its own source, and the neighbours. */
export function EntryPage(props: { readonly entry: Entry; readonly nav?: ReactNode }): ReactNode {
  const { entry } = props
  const section = SECTIONS.find((s) => s.id === entry.section)
  return (
    <>
      <PageHeader
        title={entry.question}
        eyebrow={{ label: section?.title ?? entry.section, path: `/gallery/${entry.section}` }}
        actions={props.nav}
      />
      <Prose text={entry.explain} className="-mt-5 max-w-3xl text-base leading-relaxed text-fg-secondary" />
      <LazyRecipeView id={entry.path} file={entry.file} load={entry.load} key={entry.path} />
    </>
  )
}
