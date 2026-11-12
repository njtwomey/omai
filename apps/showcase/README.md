# omai showcase

The splash screen for people who would import omai for research: short illustrated stories, each one small thing the library does, told like a
blog post. `make showcase` serves it at http://localhost:5320 (the site is `make dev`, 5300; the pocs `make pocs`, 5310). It shares the
site's colours, code block and router (imported from `apps/site/src/shell`) and nothing from `apps/pocs`.

A story is not a comparison, not a cockpit of capabilities, and has no pass/fail table. It is one idea, with a figure that earns its place.

## How a story behaves

- **Nothing happens by default.** The page loads with prose and whatever is cheap to show (the data). The reader presses a play button
  (`<Play>`); only then is the story's worker made (a module worker, fetched at that moment) and asked to run. There is no autoplay and
  no prefetch compute.
- **The computation runs in a Web Worker**, so the page stays responsive. The worker can say what it is doing in a sentence
  (`say('Warming the sampler up')`); the play button shows the last one with a calm animated indicator. There are no percentage bars.
- **The pictures come afterwards, not live.** The result is plain data; the figures and the prose that reads numbers out of it appear
  when it exists. _Run again_ re-rolls the seed (the old result stays on screen until the new one arrives).
- The worker is terminated when the run ends, when another starts and when the reader leaves the page.

## The contract (what an author writes)

A story is a folder `src/stories/<slug>/` (the slug is the folder's name and the address, `/story/<slug>`). Nothing else needs editing:
`src/registry.ts` finds it.

| File                                   | What it is                                                                                                                                                                                                                                                                                                                    |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `story.ts`                             | `export const story: StoryMeta = { title, standfirst, section, tags, date, hook? , thumbnail? }`. `section` is an id of `src/sections.ts`, the one small file that holds the order of the splash's sections.                                                                                                                  |
| `compute.ts`                           | `export const run = (seed: number, say?: Say): Result`: a pure function of the seed that returns serialisable data (it crosses the worker boundary by structured clone). The heavy imports (the engine) live here and in the worker only. The code the page shows is marked with `// region` ... `// endregion` in this file. |
| `worker.ts`                            | Always `serveStory(self, run)`.                                                                                                                                                                                                                                                                                               |
| `Story.tsx`                            | The narrative: the default export, a component. `useStoryRun` makes the run, `<Play>` is where the computation starts, figures and result-reading prose appear when `run.result` exists.                                                                                                                                      |
| `data.ts`, `Figures.tsx`, `reading.ts` | Optional: cheap data the page shows before play, the charts, and pure functions that read numbers off the result for the prose. Keep them light: they are in the page's bundle.                                                                                                                                               |
| `compute.test.ts`                      | A smoke test: the computation on a small run returns the right shapes, and the result survives `structuredClone`.                                                                                                                                                                                                             |

The kit (`src/kit/`):
- `<StoryPage>`: the layout (title, standfirst, date and tags, a single reading column, previous and next story at the end). Plain `<p>`, `<h2>`,
  `<ul>`, `<em>` and `<code>` inside it are styled; write elements, not class names. `<span className="num">` marks a number read from the result.
- `useStoryRun<Result>(createWorker)` and `<Play run label estimate />`: the run, and the button (idle with a label and an estimate; running
  with the status sentence; done with the time taken and _Run again_; error with _Try again_).
- `<Figure caption wide?>`: a figure with a caption and a short fade in; `<Reveal>`: the same fade for prose that appears after the result.
- `<Aside>`: a margin note (in the right margin on wide screens, an inset note below).
- `<Code source region? title />`: _How it is written_, from the regions of a source file imported with `?raw` (the site's `CodeBlock`
  and `snippetOf`), so the code on screen is the code that runs.

The shown code follows the house style: tensor operations only (no hand-built arrays, `for` loops or `.push`; `scripts/banned-arrays.ts` checks
`apps/showcase/src/` region by region), one variable per line, comments on their own line above the code, lines under about 100
characters. A line ending in `// hide` is not shown.

Pressing play must be the only thing that computes. Check that a story does not start a worker by being imported or rendered (the registry
test renders the pilot to a string and asserts that no worker was made), keep `Story.tsx`'s `new Worker(new URL('./worker.ts', import.meta.url),
{ type: 'module' })` literal in the file (the bundler needs to see it), and keep motion small: `prefers-reduced-motion` turns the fades and the
indicator's breathing off.

## Template

`story.ts`

```ts
import type { StoryMeta } from '../../story.ts'

export const story: StoryMeta = {
  title: 'A few words',
  standfirst: 'One line that makes you want to press play.',
  section: 'uncertainty',
  tags: ['bayesian'],
  date: '2026-11-12',
}
```

`compute.ts`

```ts
import type { Say } from '../../kit/protocol.ts'

export interface Result {
  readonly values: readonly number[]
}

export function run(seed: number, say: Say = () => {}): Result {
  say('Doing the heavy part')
  // region
  // the code the page shows goes here
  // endregion
  return { values: [seed] }
}
```

`worker.ts`

```ts
import { serveStory } from '../../kit/protocol.ts'
import { run } from './compute.ts'

serveStory(self, run)
```

`Story.tsx`

```tsx
import type { ReactNode } from 'react'
import { Code, Figure, Reveal } from '../../kit/Blocks.tsx'
import { Play } from '../../kit/Play.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'

export default function Story(): ReactNode {
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  return (
    <StoryPage>
      <p>The set-up, in prose.</p>
      <Play run={run} label="Run it" estimate="a few seconds" />
      {run.result === undefined ? null : (
        <Reveal>
          <Figure caption="What it shows.">{/* a chart of run.result */}</Figure>
          <p>What to notice, with numbers read from the result.</p>
        </Reveal>
      )}
      <Code source={source} title="compute.ts" />
    </StoryPage>
  )
}
```

## Parts

- `src/main.tsx` (the routes `/` and `/story/<slug>`), `Splash.tsx` (the hero and the sections as a grid of cards: title, one-line hook,
  a thumbnail or a tinted placeholder), `StoryRoute.tsx`, `registry.ts` (`import.meta.glob` over `stories/*/story.ts`, eager, and
  `Story.tsx`, lazy), `story.ts` (the metadata type, grouping by section, neighbours), `sections.ts`.
- `src/kit/protocol.ts` is the wire to the worker (`serveStory`, `Say`; no React, so a worker bundles only this), `runState.ts` is the run's state
  as a pure reducer, `useStoryRun.ts` the hook.
- A thumbnail is optional: put an image in `public/thumbs/` and name it in `story.ts` (`thumbnail: '/thumbs/<slug>.png'`).
