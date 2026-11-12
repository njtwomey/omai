import type { ReactNode } from 'react'
import schoolsSource from '../../../../../packages/engine/src/lib/probprog/models/schools.ts?raw'
import { snippetOf } from '../../../../site/src/shell/snippet.ts'
import { Aside, Code, Figure, Reveal } from '../../kit/Blocks.tsx'
import { Play } from '../../kit/Play.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'
import { SchoolsFigure, SpreadFigure } from './Figures.tsx'
import { readResult, SPREAD_LIMIT } from './reading.ts'

const fixed = (v: number, digits = 1): string => v.toFixed(digits)
const percent = (v: number): string => `${Math.round(v * 100)}%`

// The model the worker runs is the library's, and its source is the library's file: the pieces that make up the non-centred form.
const MODEL = `// region\n${['schools', 'schools:mu', 'schools:tau', 'schools:non-centred']
  .map((region) => snippetOf(schoolsSource, { region }))
  .join('\n// ...\n\n')}\n// endregion\n`

/** 'Eight schools, and why the average is not enough': the narrative, with the play button where the computation starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  const result = run.result
  const r = result === undefined ? undefined : readResult(result)
  return (
    <StoryPage>
      <p>
        Eight schools tried the same coaching programme for the SAT, and each reported how much it had helped. School A
        says 28 points. School C says it did harm, minus 3. The others sit in between. These are the numbers, with a bar
        of one standard error each way, because every one of them is a noisy measurement.
      </p>

      <Figure
        caption="The estimated effect of the programme at each of eight schools (dot), with one standard error either side (bar). Six of the eight bars reach zero."
        wide
      >
        <SchoolsFigure result={undefined} />
      </Figure>

      <p>
        Look at the bars. A reads 28, but its standard error is 15, so nothing here says that A really is more than
        three times better than B. Only some of the spread between the dots is the schools being different; some of it
        is just luck on each school's day. The question is how much.
      </p>

      <h2>Two answers, both wrong</h2>

      <p>
        You can take each school at its word. Then A is a triumph and C a failure, and the story is only as good as its
        noisiest measurement. Or you can ignore which school is which, average all eight, and give the same answer to
        everyone. That throws away the possibility that the programme worked better in some places than others, which is
        a thing worth knowing if it is true.
      </p>

      <p>
        The model in between says each school has its own <em>true</em> effect, and that these true effects are
        themselves drawn from one population of schools, with an unknown mean, and an unknown spread called <em>tau</em>
        . If tau turns out large, the schools really do differ, and each estimate should be left more or less alone. If
        tau turns out near zero, they are all the same, and each estimate is mostly noise around the common mean. The
        data estimate tau along with everything else, and what they say decides how hard each school is pulled towards
        the others.
      </p>

      <p>
        A school with a noisy estimate has little to say for itself, so it leans on the rest. This is called{' '}
        <em>borrowing strength</em>: it is what makes the eight estimates better together than separately.
      </p>

      <Aside>
        The model is written so that each school's effect is the common mean plus tau times a standard normal, instead
        of a normal drawn directly around the mean with spread tau. It is the same distribution. The direct way builds a
        narrow funnel when tau is small, and a sampler struggles to get into it; this way it does not.
      </Aside>

      <p>
        When you press play, the model is conditioned on the eight estimates by NUTS, which draws samples of all ten
        unknowns, the eight true effects, their mean and tau, together.
      </p>

      <Play run={run} label="Pool the schools" estimate="about five seconds" />

      {r === undefined || result === undefined ? (
        <p className="text-muted">
          Nothing has been computed yet. When you press play, the sampler runs in a background worker and the pooled
          estimates appear in the figure above.
        </p>
      ) : (
        <Reveal>
          <h2>Pulled towards the middle</h2>

          <Figure
            caption="The same estimates (orange, with one standard error either side) and, after pooling, each school's effect (blue: its average and the range holding 90% of it). The grey line joins each school's estimate to its pooled one. The dashed line is the average effect over all schools."
            wide
          >
            <SchoolsFigure result={result} />
          </Figure>

          <p>
            Every estimate has moved towards the dashed line, which is the average effect over the schools,{' '}
            <span className="num">{fixed(result.mu.mean)}</span> points. School {r.mover.name} moved furthest, from{' '}
            <span className="num">{fixed(r.mover.estimate, 0)}</span> to{' '}
            <span className="num">{fixed(r.mover.pooled.mean)}</span>, a shift of{' '}
            <span className="num">{fixed(r.moved)}</span> points. The standard deviation of the eight estimates fell
            from <span className="num">{fixed(r.rawSpread)}</span> to{' '}
            <span className="num">{fixed(r.pooledSpread)}</span>.
          </p>

          <p>
            How far each one moved depends on how noisy it was. The model pulls school {r.pulledMost.name}, whose
            standard error is <span className="num">{fixed(r.pulledMost.error, 0)}</span>, about{' '}
            <span className="num">{percent(r.pulledMost.pull)}</span> of the way to the average. School{' '}
            {r.pulledLeast.name}, with a standard error of <span className="num">{fixed(r.pulledLeast.error, 0)}</span>,
            is pulled <span className="num">{percent(r.pulledLeast.pull)}</span> of the way. The better measured a
            school, the more of its own estimate it keeps. Each school's effect now comes with a blue band, the range
            holding 90% of it, and the band still includes zero for{' '}
            <span className="num">{r.containZero === 8 ? 'all eight' : `${r.containZero} of the eight`}</span> of them.
            {r.containZero === 8 ? ' The data do not settle that any single school was helped.' : ''}
          </p>

          <h2>What the data cannot say</h2>

          <Figure
            caption={`Draws of tau, the spread of the true effects between schools. The prior on tau is a half-Cauchy with scale 5, which peaks at zero.${r.beyond > 0 ? ` ${r.beyond} ${r.beyond === 1 ? 'draw' : 'draws'} above ${SPREAD_LIMIT} ${r.beyond === 1 ? 'is' : 'are'} off the right edge.` : ''}`}
          >
            <SpreadFigure result={result} limit={SPREAD_LIMIT} />
          </Figure>

          <p>
            Everything above hangs on tau, and tau is the part the data say least about. Its middle is at about{' '}
            <span className="num">{fixed(result.tau.median)}</span>, and nine in ten of its plausible values lie between{' '}
            <span className="num">{fixed(result.tau.lo)}</span> and <span className="num">{fixed(result.tau.hi)}</span>.
            That range holds both stories: schools that are nearly identical, and schools whose true effects spread by
            about <span className="num">{fixed(result.tau.hi, 0)}</span> points. About{' '}
            <span className="num">{percent(r.smallTau)}</span> of the draws are under 5. Eight measurements this noisy
            cannot tell these apart, and the prior, a half-Cauchy with its peak at zero, leans towards the small values
            too. A different prior would give a different amount of pooling, which is worth knowing before anyone quotes
            a school&rsquo;s number.
          </p>

          <p>
            The sampler kept <span className="num">{result.draws.toLocaleString('en-GB')}</span> draws and flagged{' '}
            <span className="num">{result.divergences}</span> as divergent.
          </p>

          <h2>What to take from it</h2>

          <p>
            The average of all eight would have given one number to every school. The eight separate estimates would
            have given eight numbers that disagree for reasons that are mostly luck. The pooled answer sits between, and
            the model chose where: it is the data, through tau, that decide how much schools are allowed to differ. The
            same shape turns up anywhere there are groups with different amounts of data, from the batting averages of
            players to the radon in a county&rsquo;s houses.
          </p>

          <p>
            Press <em>Run again</em>. The sampler gets a new seed, the data stay the same, and the picture hardly moves,
            except in the tail of tau, which is the part that is hardest to pin down.
          </p>
        </Reveal>
      )}

      <Code source={MODEL} title="schools.ts, in omai-engine/probprog/models" heading="The model" />
      <Code source={source} title="compute.ts" />
    </StoryPage>
  )
}
