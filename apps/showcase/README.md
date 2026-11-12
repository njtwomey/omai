# omai showcase

The splash screen for people who would import omai for research: short illustrated stories, each one small thing the library does, told like a
blog post. `make showcase` serves it at http://localhost:5320 (the site is `make dev`, 5300; the pocs `make pocs`, 5310). It shares the
site's colours, code block and router (imported from `apps/site/src/shell`) and nothing from `apps/pocs`.

A story is not a comparison, not a cockpit of capabilities, and has no pass/fail table. It is one idea, with a figure that earns its place.

The prose is plain and in order: what happens in the order it happens. Sentences average under about 18 words. A paragraph holds
one idea and at most one new technical word, explained where it first appears. Common words, digits for numbers, one-sentence
captions, short plain titles, and nothing said twice. No jokes, asides, metaphors, rhetorical questions, cute titles or filler
('in other words', 'essentially', 'notably', 'simply'). The same goes for the worker's status sentences, which are plain
('Running the sampler').

## How a story behaves

- **Nothing happens by default.** The page loads with prose and whatever is cheap to show (the data). The reader presses a play button
  (`<Play>`); only then is the story's worker made (a module worker, fetched at that moment) and asked to run. There is no autoplay and
  no prefetch compute.
- **The computation runs in a Web Worker**, so the page stays responsive. The worker can say what it is doing in a sentence
  (`say('Running the sampler')`); the play button shows the last one with a calm animated indicator. There are no percentage bars.
- **The pictures come afterwards, not live.** The result is plain data; the figures and the prose that reads numbers out of it appear
  when it exists. _Run again_ re-rolls the seed (the old result stays on screen until the new one arrives).
- **The page reads straight down.** Pressing play changes nothing above the button. Above it is only what exists before the run: the
  text, the data, the question (a chart of the data alone is fine). Everything the run produces (the result figures, the prose that
  quotes numbers, the closing paragraphs and the code) is inside `<Results>`, placed right after `<Play>`, so it appears below the
  button in reading order and the reader never scrolls back up. A chart shown before and after (the data, then the data with the
  fitted result) is drawn twice: data alone above the button, and the result chart (which includes the data) inside `<Results>`.
  Put the button at the point in the text where the story needs the result; with two runs, use two buttons, each followed by its
  own `<Results>`.
- The worker is terminated when the run ends, when another starts and when the reader leaves the page.

## The contract (what an author writes)

A story is a folder `src/stories/<slug>/` (the slug is the folder's name and the address, `/story/<slug>`). Nothing else needs editing:
`src/registry.ts` finds it.

| File                                   | What it is                                                                                                                                                                                                                                                                                                                    |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `story.ts`                             | `export const story: StoryMeta = { title, standfirst, section, tags, date, hook?, first? }`. `section` is an id of `src/sections.ts`, the one small file that holds the order of the splash's sections.                                                                                                                       |
| `compute.ts`                           | `export const run = (seed: number, say?: Say): Result`: a pure function of the seed that returns serialisable data (it crosses the worker boundary by structured clone). The heavy imports (the engine) live here and in the worker only. The code the page shows is marked with `// region` ... `// endregion` in this file. |
| `worker.ts`                            | Always `serveStory(self, run)`.                                                                                                                                                                                                                                                                                               |
| `Story.tsx`                            | The narrative: the default export, a component. `useStoryRun` makes the run, `<Play>` is where the computation starts, and `<Results run={run}>` right after it holds everything the result makes (figures, result-reading prose, closing prose, code).                                                                       |
| `data.ts`, `Figures.tsx`, `reading.ts` | Optional: cheap data the page shows before play, the charts, and pure functions that read numbers off the result for the prose. Keep them light: they are in the page's bundle.                                                                                                                                               |
| `compute.test.ts`                      | A smoke test: the computation on a small run returns the right shapes, and the result survives `structuredClone`.                                                                                                                                                                                                             |

The kit (`src/kit/`):
- `<StoryPage>`: the layout (title, standfirst, date and tags, a single reading column, previous and next story at the end). Plain `<p>`, `<h2>`,
  `<ul>`, `<em>` and `<code>` inside it are styled; write elements, not class names. `<span className="num">` marks a number read from the result.
- `useStoryRun<Result>(createWorker)` and `<Play run label estimate />`: the run, and the button (idle with a label and an estimate; running
  with the status sentence; done with the time taken and _Run again_; error with _Try again_).
- `<Results run={run}>{(result) => ...}</Results>`: place it right after `<Play>`. While there is no result it shows one muted line
  ('Press play. The result will appear here.') and builds nothing; when the result exists it calls the function
  with it and shows the content with the kit's fade (figures and `<Reveal>` inside it do not fade again). Put the closing prose and
  the `<Code>` blocks inside it too: with nothing below the block, nothing the reader is looking at can move when the result
  arrives. When the result arrives and the block's top is below the fold, the page scrolls it into view (smoothly, instantly under
  `prefers-reduced-motion`); otherwise it never scrolls. The registry test checks every story: a play button, then a results block,
  empty on the first render, and no code before it.
- `<Choice label options value onChange />`: a small radio group (a label, and for each choice a name with a one-line description under
  it; arrow keys move between the choices; the choices stack at a narrow width) for the rare story whose reader decides how
  something is built before pressing play. The value lives in the story's own `useState`; to clear the result when a choice
  changes, render the play button and the results in a child with `key={choice}` (its `useStoryRun` then starts afresh), and send
  the choice to the worker with a first `postMessage` in `createWorker` (see `gaussian-mixtures`). It is the one exception to
  "no controls", made at the owner's request.
- `<Figure caption wide?>`: a figure with a caption and a short fade in; `<Reveal>`: the same fade for prose that appears after the result.
- `<Tex>` and `<Equation>` (`src/kit/Maths.tsx`): maths, typeset by omai-render's TeX module (KaTeX with the one shared macro registry:
  `\Gauss`, `\expect`, `\KL`, `\xvec`, `\sigmoid`, ...; its stylesheet and fonts come with `omai-render/react`, so there is nothing to
  load; a macro a story needs is added to the registry in `packages/render/src/maths/macros.ts`, never defined in the app). `<Tex>` is
  inline, for the symbol of a quantity or a short formula in a sentence (`J`, `\mu`, `\sigma^2`). `<Equation label?>` is a display
  equation, centred in the column with a figure's spacing, scrolling sideways inside its own box when a line is too wide. Write the sentence
  first and the equation after it. Numbers read from the result go in the slots of the `tex` tag, which sets them like `.num`:
  `<Tex>{tex`J = ${fixed(precision)}`}</Tex>`; constants of the model are written in the TeX itself. Both carry a text alternative for
  screen readers (the TeX, or a `reading` in words, which an `Equation` should have). Keep `<code>` for code names (`damping`,
  `packScalars`). A message on an edge of `<MessageReplay>` is typeset the same way.
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
import { Code, Figure } from '../../kit/Blocks.tsx'
import { Equation, Tex, tex } from '../../kit/Maths.tsx'
import { Play } from '../../kit/Play.tsx'
import { Results } from '../../kit/Results.tsx'
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
      <Results run={run}>
        {(result) => (
          <>
            <Figure caption="What it shows.">{/* a chart of result */}</Figure>
            <p>
              What to notice, with numbers read from the result: the mean is{' '}
              <Tex>{tex`\mu = ${fixed(result.mean)}`}</Tex>.
            </p>
            <Equation label="The model">{String.raw`y_i \sim \Gauss(\mu, \sigma^2)`}</Equation>
            <p>How it ends, and what to read next.</p>
            <Code source={source} title="compute.ts" />
          </>
        )}
      </Results>
    </StoryPage>
  )
}
```

## Parts

- `src/main.tsx` (the routes `/` and `/story/<slug>`), `Splash.tsx` (the hero and the sections as a plain list: title and one-line hook,
  the list), `StoryRoute.tsx`, `registry.ts` (`import.meta.glob` over `stories/*/story.ts`, eager, and
  `Story.tsx`, lazy), `story.ts` (the metadata type, grouping by section, neighbours), `sections.ts`.
- `src/kit/protocol.ts` is the wire to the worker (`serveStory`, `Say`; no React, so a worker bundles only this), `runState.ts` is the run's state
  as a pure reducer, `useStoryRun.ts` the hook.
- `first: true` lists a story before the others of its section (the way "How expectation propagation works" opens its category).
