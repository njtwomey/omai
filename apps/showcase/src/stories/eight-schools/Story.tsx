import type { ReactNode } from 'react'
import schoolsSource from '../../../../../packages/engine/src/lib/probprog/models/schools.ts?raw'
import { snippetOf } from '../../../../site/src/shell/snippet.ts'
import { Aside, Code, Figure } from '../../kit/Blocks.tsx'
import { Play } from '../../kit/Play.tsx'
import { Results } from '../../kit/Results.tsx'
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

/** 'Eight schools and partial pooling': the narrative, with the play button where the computation starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  return (
    <StoryPage>
      <p>
        Eight schools tried the same coaching programme for the SAT. Each school reported how much the programme helped
        its students, in points. School A reports 28 points. School C reports −3, which means the programme did harm.
        The other schools lie in between. Every report is a noisy estimate, so the figure shows each one with a bar of
        one standard error on each side.
      </p>

      <Figure
        caption="The estimated effect of the programme at each of eight schools (dot), with one standard error either side (bar). Six of the eight bars reach zero."
        wide
      >
        <SchoolsFigure result={undefined} />
      </Figure>

      <p>
        School A reads 28, but its standard error is 15. So the data do not show that A&rsquo;s programme worked three
        times better than B&rsquo;s. Part of the spread between the dots is real difference between the schools. Part is
        chance in each school&rsquo;s estimate. The question is how much of each.
      </p>

      <h2>Two simple answers</h2>

      <p>
        One answer is to take each estimate as it is. Then A is a success and C is a failure, and each conclusion rests
        on one noisy number. The other answer is to average all eight and give the same value to every school. That
        ignores any real difference between the schools.
      </p>

      <p>
        A model in between works like this. Each school has its own true effect. The true effects come from one
        population of schools, with an unknown mean and an unknown spread. The spread is called <em>tau</em>. If tau is
        large, the schools really differ, and each estimate should be left mostly alone. If tau is near zero, the
        schools are the same, and each estimate is the common mean plus noise. The data estimate tau along with
        everything else. Its value decides how far each school is pulled towards the others.
      </p>

      <p>
        A school with a noisy estimate has little to say for itself, so it relies more on the others. This is called{' '}
        <em>borrowing strength</em>.
      </p>

      <Aside>
        The model writes each school&rsquo;s effect as the common mean plus tau times a standard normal. This is the
        same distribution as a normal centred on the mean with spread tau. The direct form makes a narrow funnel when
        tau is small, which is hard for a sampler to enter. The form used here does not.
      </Aside>

      <p>
        We condition the model on the eight estimates with NUTS. It draws samples of all ten unknowns together: the
        eight true effects, their mean and tau. Press play to run it.
      </p>

      <Play run={run} label="Pool the schools" estimate="about five seconds" />

      <Results run={run}>
        {(result) => {
          const r = readResult(result)
          return (
            <>
              <h2>Pulled towards the middle</h2>

              <p>The figure shows each school&rsquo;s estimate again, and its effect after pooling.</p>

              <Figure
                caption="The estimates (orange, with one standard error either side) and, after pooling, each school's effect (blue: its average and the range holding 90% of it). The grey line joins each school's estimate to its pooled one. The dashed line is the average effect over all schools."
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
                How far a school moves depends on how noisy its estimate is. School {r.pulledMost.name}, with a standard
                error of <span className="num">{fixed(r.pulledMost.error, 0)}</span>, is pulled about{' '}
                <span className="num">{percent(r.pulledMost.pull)}</span> of the way to the average. School{' '}
                {r.pulledLeast.name}, with a standard error of{' '}
                <span className="num">{fixed(r.pulledLeast.error, 0)}</span>, is pulled{' '}
                <span className="num">{percent(r.pulledLeast.pull)}</span> of the way. The better a school is measured,
                the more of its own estimate it keeps.
              </p>

              <p>
                Each pooled effect has a blue band that holds 90% of its draws. The band includes zero for{' '}
                <span className="num">{r.containZero === 8 ? 'all eight' : `${r.containZero} of the eight`}</span>{' '}
                schools.
                {r.containZero === 8 ? ' The data do not show that any single school was helped.' : ''}
              </p>

              <h2>What the data cannot say</h2>

              <p>All of this depends on tau. The next figure shows the draws of tau.</p>

              <Figure
                caption={`Draws of tau, the spread of the true effects between schools. The prior on tau is a half-Cauchy with scale 5, which peaks at zero.${r.beyond > 0 ? ` ${r.beyond} ${r.beyond === 1 ? 'draw' : 'draws'} above ${SPREAD_LIMIT} ${r.beyond === 1 ? 'is' : 'are'} off the right edge.` : ''}`}
              >
                <SpreadFigure result={result} limit={SPREAD_LIMIT} />
              </Figure>

              <p>
                The middle of tau is about <span className="num">{fixed(result.tau.median)}</span>. Nine in ten of its
                draws lie between <span className="num">{fixed(result.tau.lo)}</span> and{' '}
                <span className="num">{fixed(result.tau.hi)}</span>. That range holds two cases: schools that are nearly
                identical, and schools whose true effects spread by about{' '}
                <span className="num">{fixed(result.tau.hi, 0)}</span> points. About{' '}
                <span className="num">{percent(r.smallTau)}</span> of the draws are under 5. Eight estimates this noisy
                cannot tell these cases apart. The prior also leans towards small values, because a half-Cauchy has its
                peak at zero. A different prior would give a different amount of pooling.
              </p>

              <p>
                The sampler kept <span className="num">{result.draws.toLocaleString('en-GB')}</span> draws and flagged{' '}
                <span className="num">{result.divergences}</span> as divergent.
              </p>

              <h2>What to take from it</h2>

              <p>
                Averaging all eight would give one number to every school. Reading the eight separately would give eight
                numbers that differ mostly by chance. The pooled answer lies between the two. The data, through tau,
                decide how much the schools may differ. The same structure appears wherever groups have different
                amounts of data, such as the batting averages of players or the radon levels in the houses of different
                counties.
              </p>

              <p>
                Press <em>Run again</em> to use a new seed. The data stay the same, and the picture changes little,
                except in the tail of tau, which is the hardest part to pin down.
              </p>

              <Code source={MODEL} title="schools.ts, in omai-engine/probprog/models" heading="The model" />
              <Code source={source} title="compute.ts" />
            </>
          )
        }}
      </Results>
    </StoryPage>
  )
}
