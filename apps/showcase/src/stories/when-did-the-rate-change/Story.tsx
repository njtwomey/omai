import type { ReactNode } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Aside, Code, Figure, Reveal } from '../../kit/Blocks.tsx'
import { Play } from '../../kit/Play.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import { storyHref } from '../../story.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'
import { coalCounts, SIMULATED, yearOf } from './data.ts'
import { CompareFigure, SeriesFigure, WhenFigure } from './Figures.tsx'
import { MASS, readData, readResult } from './reading.ts'

// The real counts are a table lookup, so the page shows them before anything runs.
const COUNTS = coalCounts()
const DATA = readData(COUNTS)

const fixed = (v: number, digits = 1): string => v.toFixed(digits)
const percent = (v: number): string => `${Math.round(100 * v)}%`

/** 'When did the rate change?': the narrative, with the play button where the message passing starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  const result = run.result
  const r = result === undefined ? undefined : readResult(result)
  return (
    <StoryPage>
      <p>
        Between <span className="num">{DATA.firstYear}</span> and <span className="num">{DATA.lastYear}</span>,{' '}
        <span className="num">{DATA.total}</span> explosions in British coal mines killed ten or more men. Here they
        are, counted by year. The bars are tall on the left and short on the right, so something changed. The
        detective&rsquo;s question is when.
      </p>

      <Figure
        caption={
          result === undefined
            ? 'Disasters per year in British coal mines, 1851 to 1962.'
            : 'The same counts, with the rate the model expects in each year (line): the old rate, then the new one, with a short blend where the date is in doubt.'
        }
        wide
      >
        <SeriesFigure counts={COUNTS} detection={result?.coal} />
      </Figure>

      <p>
        It is tempting to point at a year. But a rate is not a count. Even with the rate unchanged, some years bring
        none and some bring five, so a quiet year in the middle of a bad stretch proves nothing, and neither does one
        bad year in a quiet stretch. We never see the rate itself, only the counts it throws up, and the counts jitter.
      </p>

      <Aside>
        Counts of rare events in a fixed time, with a steady underlying rate, follow a Poisson distribution: a rate of 3
        a year gives 3 on average, but plenty of years with 1 and with 6.
      </Aside>

      <h2>A model with one switch</h2>

      <p>
        So we ask the question a different way. Suppose there are two rates, an old one and a new one, both unknown.
        Each year is in one of two phases, &ldquo;before&rdquo; or &ldquo;after&rdquo;, and its count comes from the
        rate of its phase. The phase is hidden. A year can only move from before to after and never back, and there is a
        small, steady chance, about one in {DATA.years}, that any given year is the one in which it happens. That is the
        whole model.
      </p>

      <p>
        The unknown we care about is a discrete one: which of the {DATA.years} years is the first of the new regime.
        Most of the model&rsquo;s unknowns are numbers on a scale; this one is a position on a calendar. Messages handle
        it by letting each year tell its neighbours how likely it is that the switch is behind them, and by letting the
        two rates tell every year how well its count fits them. They pass those notes along the years and back until
        none of them changes.
      </p>

      <Play run={run} label="Find the year" estimate="about 2 seconds, in the background" />

      {r === undefined || result === undefined ? (
        <p className="text-muted">
          Nothing has been computed yet. When you press play, the messages are passed in a background worker. The
          expected rate appears on the figure above and the chance of each year below it.
        </p>
      ) : (
        <Reveal>
          <h2>Not a date, a distribution</h2>

          <Figure caption="The chance that the switch happened in each year. It is a distribution over a calendar, drawn on the same years as the counts above.">
            <WhenFigure detection={result.coal} />
          </Figure>

          <p>
            The most likely year is <span className="num">{yearOf(r.coal.likeliest)}</span>, but it holds only{' '}
            <span className="num">{percent(r.coal.likeliestChance)}</span> of the chance. The shortest run of years that
            holds {percent(MASS)} of it is {yearOf(r.coal.window.from)} to {yearOf(r.coal.window.to)}, which is{' '}
            <span className="num">{r.coal.window.years}</span> years. The honest answer to &ldquo;when did the rate
            change?&rdquo; is therefore not a year. It is a shape, with a peak and some neighbours.
          </p>

          <p>
            The two rates come out as well. Before the switch the model expects about{' '}
            <span className="num">{fixed(result.coal.before.mean)}</span> disasters a year, with nine in ten of the
            plausible rates between <span className="num">{fixed(result.coal.before.lo)}</span> and{' '}
            <span className="num">{fixed(result.coal.before.hi)}</span>. After it, about{' '}
            <span className="num">{fixed(result.coal.after.mean, 2)}</span>, between{' '}
            <span className="num">{fixed(result.coal.after.lo, 2)}</span> and{' '}
            <span className="num">{fixed(result.coal.after.hi, 2)}</span>. That is a fall by a factor of about{' '}
            <span className="num">{fixed(r.coal.drop)}</span>. The data do not say why it fell; they say only that it
            did, and roughly when.
          </p>

          <h2>The sharper the change, the sharper the date</h2>

          <p>
            Why is the answer a span of <span className="num">{r.coal.window.years}</span> years and not two or twenty?
            It depends on how big the change was. To see it, here is the same model run on two made-up series, drawn
            fresh for this run. Both are as long as the real one and both change in year {SIMULATED.at + 1}. In the
            first the rate falls from <span className="num">{SIMULATED.abrupt.before}</span> a year to{' '}
            <span className="num">{SIMULATED.abrupt.after}</span>, an abrupt drop. In the second it falls only from{' '}
            <span className="num">{SIMULATED.gentle.before}</span> to{' '}
            <span className="num">{SIMULATED.gentle.after}</span>.
          </p>

          <Figure caption="Made-up data, with the true switch marked. The model sees only the counts. The abrupt change gives a spike; the gentle one a broad hill.">
            <CompareFigure result={result} />
          </Figure>

          <p>
            {r.abruptSharper ? (
              <>
                The abrupt series pins the switch down to <span className="num">{r.abrupt.window.years}</span> years,
                the gentle one only to <span className="num">{r.gentle.window.years}</span>.
              </>
            ) : (
              <>
                This time the abrupt series holds its {percent(MASS)} within{' '}
                <span className="num">{r.abrupt.window.years}</span> years and the gentle one within{' '}
                <span className="num">{r.gentle.window.years}</span>, so the two are closer than they usually are.
              </>
            )}{' '}
            The true year, {SIMULATED.at + 1}, lies {r.abruptCovers ? 'inside' : 'outside'} the window of the first and{' '}
            {r.gentleCovers ? 'inside' : 'outside'} the window of the second. A big change leaves little doubt about
            when it came; a small one could have come at many times and the counts cannot tell them apart. The
            coal-mining record sits between the two: the rate fell by about{' '}
            <span className="num">{fixed(r.coal.drop)}</span> times, where the abrupt series falls by{' '}
            <span className="num">{fixed(r.abrupt.drop)}</span> and the gentle one by{' '}
            <span className="num">{fixed(r.gentle.drop, 2)}</span>.
          </p>

          <h2>Why this is worth knowing</h2>

          <p>
            A changepoint is a good example of an unknown that is not a number but a choice, here among {DATA.years}{' '}
            calendar years. A tool that gives you a single date has thrown away the part that matters for deciding what
            to do about it. A distribution over the date says how much to trust a story like &ldquo;it changed around
            then&rdquo;, and anyone checking whether a new rule made the difference can ask whether it came inside the
            window or outside it.
          </p>

          <p>
            Press <em>Run again</em> for two new made-up series. The coal-mining answer will not move, because nothing
            in that part of the run is random: the same counts and the same notes give the same distribution every time.
            For another story told with message passing, see{' '}
            <Link to={storyHref('two-humps')} className="underline underline-offset-2">
              Two humps
            </Link>
            .
          </p>
        </Reveal>
      )}

      <Code source={source} title="compute.ts" />
    </StoryPage>
  )
}
