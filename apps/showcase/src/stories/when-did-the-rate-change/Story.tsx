import type { ReactNode } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Aside, Code, Figure } from '../../kit/Blocks.tsx'
import { Play } from '../../kit/Play.tsx'
import { Results } from '../../kit/Results.tsx'
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
  return (
    <StoryPage>
      <p>
        Between <span className="num">{DATA.firstYear}</span> and <span className="num">{DATA.lastYear}</span>,{' '}
        <span className="num">{DATA.total}</span> explosions in British coal mines killed ten or more men. The figure
        counts them by year. The bars are tall on the left and short on the right, so the rate of disasters changed. The
        question is when.
      </p>

      <Figure caption={`Disasters per year in British coal mines, ${DATA.firstYear} to ${DATA.lastYear}.`} wide>
        <SeriesFigure counts={COUNTS} detection={undefined} />
      </Figure>

      <p>
        It is tempting to point at a year. But a rate is not a count. Even at a steady rate, some years have none and
        some have five. A quiet year in a bad stretch proves nothing, and neither does a bad year in a quiet stretch. We
        never see the rate itself, only the counts, and the counts vary.
      </p>

      <Aside>
        Counts of rare events in a fixed time, at a steady rate, follow a Poisson distribution. A rate of 3 a year gives
        3 on average, but many years with 1 and many with 6.
      </Aside>

      <h2>A model with one switch</h2>

      <p>
        So we ask the question another way. Suppose there are two rates, an old one and a new one, both unknown. Each
        year is in one of two phases, before or after, and its count comes from the rate of its phase. The phase is
        hidden. It can change from before to after, never back. Any given year has a small, steady chance, about one in{' '}
        {DATA.years}, of being the year it changes. That is the whole model.
      </p>

      <p>
        The unknown we care about is the year of the switch: which of the {DATA.years} years is the first of the new
        rate. Expectation propagation finds it with messages. Each year tells its neighbours how likely it is that the
        switch is behind them. The two rates tell each year how well its count fits them. The messages pass along the
        years and back until none of them changes.
      </p>

      <p>Press play to run the message passing.</p>

      <Play run={run} label="Find the year" estimate="about 2 seconds, in the background" />

      <Results run={run}>
        {(result) => {
          const r = readResult(result)
          return (
            <>
              <h2>The expected rate</h2>

              <p>The model gives a rate for each year. The figure draws it over the counts.</p>

              <Figure
                caption="The counts again, with the rate the model expects in each year (line): the old rate, a short blend, then the new rate."
                wide
              >
                <SeriesFigure counts={COUNTS} detection={result.coal} />
              </Figure>

              <h2>Not a date, a distribution</h2>

              <p>The model also gives the chance that the switch happened in each year.</p>

              <Figure caption="The chance that the switch happened in each year, on the same years as the counts above.">
                <WhenFigure detection={result.coal} />
              </Figure>

              <p>
                The most likely year is <span className="num">{yearOf(r.coal.likeliest)}</span>, but it holds only{' '}
                <span className="num">{percent(r.coal.likeliestChance)}</span> of the chance. The shortest run of years
                that holds {percent(MASS)} of it is {yearOf(r.coal.window.from)} to {yearOf(r.coal.window.to)}, which is{' '}
                <span className="num">{r.coal.window.years}</span> years. So the answer is not one year. It is a
                distribution, with a peak and some neighbours.
              </p>

              <p>
                The two rates come out as well. Before the switch the model expects about{' '}
                <span className="num">{fixed(result.coal.before.mean)}</span> disasters a year, with nine in ten of the
                plausible rates between <span className="num">{fixed(result.coal.before.lo)}</span> and{' '}
                <span className="num">{fixed(result.coal.before.hi)}</span>. After it, the model expects about{' '}
                <span className="num">{fixed(result.coal.after.mean, 2)}</span>, between{' '}
                <span className="num">{fixed(result.coal.after.lo, 2)}</span> and{' '}
                <span className="num">{fixed(result.coal.after.hi, 2)}</span>. That is a fall by a factor of about{' '}
                <span className="num">{fixed(r.coal.drop)}</span>. The data do not say why it fell, only that it did,
                and roughly when.
              </p>

              <h2>The sharper the change, the sharper the date</h2>

              <p>
                Why is the answer a span of <span className="num">{r.coal.window.years}</span> years, and not two or
                twenty? It depends on how big the change was. To see this, we run the same model on two made-up series,
                drawn for this run. Both are as long as the real one, and both change in year {SIMULATED.at + 1}.
              </p>

              <p>
                In the first, the rate falls from <span className="num">{SIMULATED.abrupt.before}</span> a year to{' '}
                <span className="num">{SIMULATED.abrupt.after}</span>, an abrupt drop. In the second, it falls only from{' '}
                <span className="num">{SIMULATED.gentle.before}</span> to{' '}
                <span className="num">{SIMULATED.gentle.after}</span>.
              </p>

              <Figure caption="Made-up data, with the true switch marked. The abrupt change gives a spike and the gentle change gives a broad hill.">
                <CompareFigure result={result} />
              </Figure>

              <p>
                {r.abruptSharper ? (
                  <>
                    The abrupt series pins the switch down to <span className="num">{r.abrupt.window.years}</span>{' '}
                    years. The gentle one pins it down only to <span className="num">{r.gentle.window.years}</span>.
                  </>
                ) : (
                  <>
                    This time the abrupt series holds its {percent(MASS)} within{' '}
                    <span className="num">{r.abrupt.window.years}</span> years and the gentle one within{' '}
                    <span className="num">{r.gentle.window.years}</span>, so the two are closer than they usually are.
                  </>
                )}{' '}
                The true year, {SIMULATED.at + 1}, lies {r.abruptCovers ? 'inside' : 'outside'} the window of the first
                and {r.gentleCovers ? 'inside' : 'outside'} the window of the second.
              </p>

              <p>
                A big change leaves little doubt about when it came. A small change could have come at many times, and
                the counts cannot tell them apart. The coal-mining rate fell by a factor of{' '}
                <span className="num">{fixed(r.coal.drop)}</span>, the abrupt series by{' '}
                <span className="num">{fixed(r.abrupt.drop)}</span> and the gentle one by{' '}
                <span className="num">{fixed(r.gentle.drop, 2)}</span>.
              </p>

              <h2>What to take from it</h2>

              <p>
                A changepoint is an unknown that is not a number but a choice, here among {DATA.years} calendar years. A
                single date throws away the part that matters for deciding what to do. A distribution over the date says
                how much to trust &ldquo;it changed around then&rdquo;. To check whether a new rule made the difference,
                ask whether it came inside the window or outside it.
              </p>

              <p>
                Press <em>Run again</em> for two new made-up series. The coal-mining answer does not change, because
                nothing in that part of the run is random. For another story told with message passing, see{' '}
                <Link to={storyHref('two-humps')} className="underline underline-offset-2">
                  Two kinds of wait at Old Faithful
                </Link>
                .
              </p>

              <Code source={source} title="compute.ts" />
            </>
          )
        }}
      </Results>
    </StoryPage>
  )
}
