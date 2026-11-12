import type { ReactNode } from 'react'
import schoolsSource from '../../../../../packages/engine/src/lib/probprog/models/schools.ts?raw'
import { snippetOf } from '../../../../site/src/shell/snippet.ts'
import { Aside, Code, Figure } from '../../kit/Blocks.tsx'
import { Equation, Tex } from '../../kit/Maths.tsx'
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
        Eight schools tried the same coaching programme for the SAT. Each reported how many points the programme helped
        its students. School A reports 28 points. School C reports −3, so the programme did harm. The other schools lie
        in between. Every report is a noisy estimate, so the figure shows each with a bar of one standard error (a
        measure of its noise) on each side.
      </p>

      <Figure
        caption="The estimated effect at each school (dot), with one standard error either side (bar). Six of the eight bars reach zero."
        wide
      >
        <SchoolsFigure result={undefined} />
      </Figure>

      <p>
        School A reads 28, but its standard error is 15, so the data do not show that A&rsquo;s programme worked three
        times better than B&rsquo;s. Part of the spread between the dots is real difference between schools, and part is
        chance. The question is how much of each.
      </p>

      <h2>Two simple answers</h2>

      <p>
        One answer is to take each estimate as it is. Then A is a success and C is a failure, each on one noisy number.
        The other is to average all eight and give that value to every school, which ignores any real difference.
      </p>

      <p>
        A model in between works like this. Each school has its own true effect. The true effects come from one
        population of schools, with an unknown mean <Tex>\mu</Tex> and an unknown spread <Tex>\tau</Tex>.
      </p>

      <p>
        If <Tex>\tau</Tex> is large, the schools really differ, and each estimate stays mostly alone. If <Tex>\tau</Tex>{' '}
        is near zero, the schools are the same. The data estimate <Tex>\tau</Tex> with everything else, and{' '}
        <Tex>\tau</Tex> decides how far each school is pulled towards the others. A school with a noisy estimate has
        little to say for itself, so it relies more on the others. This is called <em>borrowing strength</em>.
      </p>

      <Equation
        label={
          <>
            The model, with <Tex>\sigma_j</Tex> the standard error of school <Tex>j</Tex>
          </>
        }
        reading="The mean mu is normal with mean 0 and spread 5. The spread tau is half-Cauchy with scale 5. The effect theta j of school j is mu plus tau times a standard normal z j. The estimate y j is normal around theta j with spread sigma j."
      >
        {String.raw`\begin{aligned}
          \mu &\sim \Gauss(0, 5^2), \qquad \tau \sim \HalfCauchy(5) \\
          \theta_j &= \mu + \tau\, z_j, \qquad z_j \sim \Gauss(0, 1) \\
          y_j &\sim \Gauss(\theta_j, \sigma_j^2)
        \end{aligned}`}
      </Equation>

      <Aside>
        The model writes each school&rsquo;s effect as the common mean plus <Tex>\tau</Tex> times a standard normal.
        That is the same distribution as a normal with spread <Tex>\tau</Tex>, and it avoids a narrow funnel at small{' '}
        <Tex>\tau</Tex> that samplers struggle to enter.
      </Aside>

      <p>
        We condition the model on the eight estimates with NUTS. It draws all ten unknowns together: the eight true
        effects, their mean and <Tex>\tau</Tex>. Press play to run it.
      </p>

      <Play run={run} label="Pool the schools" estimate="about five seconds" />

      <Results run={run}>
        {(result) => {
          const r = readResult(result)
          return (
            <>
              <h2>Pulled towards the middle</h2>

              <Figure
                caption="Estimates (orange, with one standard error) and pooled effects (blue: average and middle 90%), joined by grey lines. The dashed line is the average effect."
                wide
              >
                <SchoolsFigure result={result} />
              </Figure>

              <p>
                Every estimate has moved towards the dashed line, the average effect over the schools,{' '}
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
                Each blue band holds 90% of a school&rsquo;s draws. The band includes zero for{' '}
                <span className="num">{r.containZero === 8 ? 'all eight' : `${r.containZero} of the eight`}</span>{' '}
                schools.
                {r.containZero === 8 ? ' The data do not show that any single school was helped.' : ''}
              </p>

              <h2>What the data cannot say</h2>

              <p>
                All of this depends on <Tex>\tau</Tex>. Here are its draws.
              </p>

              <Figure
                caption={
                  <>
                    Draws of <Tex>\tau</Tex>, the spread of the true effects. Its prior is a half-Cauchy with scale 5,
                    which peaks at zero.
                    {r.beyond > 0
                      ? ` ${r.beyond} ${r.beyond === 1 ? 'draw' : 'draws'} above ${SPREAD_LIMIT} ${r.beyond === 1 ? 'is' : 'are'} off the right edge.`
                      : ''}
                  </>
                }
              >
                <SpreadFigure result={result} limit={SPREAD_LIMIT} />
              </Figure>

              <p>
                The middle of <Tex>\tau</Tex> is about <span className="num">{fixed(result.tau.median)}</span>. Nine in
                ten of its draws lie between <span className="num">{fixed(result.tau.lo)}</span> and{' '}
                <span className="num">{fixed(result.tau.hi)}</span>. That range holds two cases: schools that are nearly
                identical, and schools whose true effects spread by about{' '}
                <span className="num">{fixed(result.tau.hi, 0)}</span> points. Eight estimates this noisy cannot tell
                them apart.
              </p>

              <p>
                About <span className="num">{percent(r.smallTau)}</span> of the draws are under 5. The prior leans that
                way too, because a half-Cauchy peaks at zero. A different prior would give a different amount of
                pooling. The sampler kept <span className="num">{result.draws.toLocaleString('en-GB')}</span> draws and
                flagged <span className="num">{result.divergences}</span> as divergent.
              </p>

              <h2>What to take from it</h2>

              <p>
                Averaging all eight gives one number to every school. Reading them separately gives eight numbers that
                differ mostly by chance. The pooled answer lies between the two, and the data, through <Tex>\tau</Tex>,
                decide how much the schools may differ. The same structure appears wherever groups have different
                amounts of data.
              </p>

              <p>
                Press <em>Run again</em> to use a new seed. The data stay the same, and the picture changes little,
                except in the tail of <Tex>\tau</Tex>, which is the hardest part to pin down.
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
