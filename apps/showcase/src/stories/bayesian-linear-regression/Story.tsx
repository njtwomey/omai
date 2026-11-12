import type { ReactNode } from 'react'
import { Aside, Code, Figure } from '../../kit/Blocks.tsx'
import { Equation, Tex, tex } from '../../kit/Maths.tsx'
import { Play } from '../../kit/Play.tsx'
import { Results } from '../../kit/Results.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'
import { simulate, TRUTH } from './data.ts'
import { CloudFigure, LinesFigure } from './Figures.tsx'
import { readResult } from './reading.ts'

// The data are twelve fixed points, drawn here so the page shows them before anything runs.
const DATA = simulate()

const fixed = (v: number, digits = 2): string => v.toFixed(digits)

/** 'A line, and how sure we are': the narrative, with the play button where the computation starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  return (
    <StoryPage>
      <p>
        Here are twelve measurements. For each one, we set a value <Tex>x</Tex> and read off a value <Tex>y</Tex>. The
        points rise from left to right, but none lies exactly on a line.
      </p>

      <Figure
        caption={
          <>
            The twelve points, between <Tex>x = 0.3</Tex> and <Tex>x = 4.3</Tex>.
          </>
        }
        wide
      >
        <LinesFigure data={DATA} result={undefined} />
      </Figure>

      <p>
        The usual step is to fit one line, <Tex>y = a + b\,x</Tex>: the intercept <Tex>a</Tex> and slope <Tex>b</Tex>{' '}
        that pass closest to the points. One line does not say how far to trust it. Many other lines fit almost as well.
      </p>

      <h2>Many lines instead of one</h2>

      <p>
        Bayesian regression keeps every line that fits. It starts from a belief about each number. The intercept and
        slope each get a normal distribution with mean 0 and spread 5. The noise <Tex>\sigma</Tex> is how far the points
        scatter around the line. It cannot be negative, so it gets a half-normal distribution with spread 2.
      </p>

      <Equation
        label="The model"
        reading="The intercept a and the slope b each have a normal prior with mean 0 and spread 5. The noise sigma has a half-normal prior with spread 2. Each y is normal around a plus b times x, with spread sigma."
      >
        {String.raw`\begin{aligned}
          a, b &\sim \Gauss(0, 5^2) \qquad \sigma \sim \HalfNormal(2) \\
          y_i &\sim \Gauss(a + b\,x_i,\; \sigma^2)
        \end{aligned}`}
      </Equation>

      <p>
        Then it uses the data. A line close to many points becomes more likely, and a line that misses them less likely.
        The result is a distribution over the intercept, slope and noise, called the <em>posterior</em>. One draw from
        it is one line.
      </p>

      <p>
        We cannot write the posterior as a formula, so we sample it with NUTS. NUTS takes steps through the intercept,
        slope and noise, guided by the gradient of the log probability. Each point it visits is one line.
      </p>

      <Aside>omai computes that gradient by automatic differentiation, so the model does not supply one.</Aside>

      <p>Press play to draw the lines.</p>

      <Play run={run} label="Draw the lines" estimate="a few seconds" />

      <Results run={run}>
        {(result) => {
          const r = readResult(result, DATA)
          return (
            <>
              <h2>The lines</h2>

              <p>
                The sampler kept {result.draws.toLocaleString('en-GB')} draws, each one a line. The figure shows{' '}
                {result.fan.length} of them. The solid line is their average, and the band holds the middle 90% at each{' '}
                <Tex>x</Tex>.
              </p>

              <Figure
                caption="Sampled lines (thin), their average (solid), the middle 90% (shaded) and the line the data came from (dashed)."
                wide
              >
                <LinesFigure data={DATA} result={result} />
              </Figure>

              <p>
                Where the lines are close together, the data fix them. Near <Tex>{tex`x = ${fixed(r.centre, 1)}`}</Tex>,
                in the middle of the points, the band is <span className="num">{fixed(r.centreWidth)}</span> wide. At{' '}
                <Tex>{tex`x = ${fixed(r.edge, 0)}`}</Tex>, far right of the last point, it is{' '}
                <span className="num">{fixed(r.edgeWidth)}</span> wide, about{' '}
                <span className="num">{fixed(r.ratio, 1)}</span> times as wide. There are no measurements there, so the
                lines disagree.
              </p>

              <Figure caption="Each dot is one sampled line, placed by its intercept and slope. The marked dot is the true line.">
                <CloudFigure result={result} />
              </Figure>

              <p>
                The dots form a tilted cloud. A steeper line needs a lower intercept to stay near the points. This is
                why the lines pivot about the middle of the data: there, a change of slope moves a line little, and far
                away it moves a line a lot.
              </p>

              <h2>What to take from it</h2>

              <p>
                A single fit gives one slope. The posterior also gives the doubt. The slope has a mean of{' '}
                <span className="num">{fixed(r.slope.mean)}</span>, and nine in ten draws lie between{' '}
                <span className="num">{fixed(r.slope.lo)}</span> and <span className="num">{fixed(r.slope.hi)}</span>.
                The true slope, {TRUTH.slope}, is {r.slopeInside ? 'inside' : 'outside'} that range.
              </p>

              <p>
                The noise has a mean of <span className="num">{fixed(r.noise.mean)}</span>, and nine in ten draws lie
                between <span className="num">{fixed(r.noise.lo)}</span> and{' '}
                <span className="num">{fixed(r.noise.hi)}</span>. The true value is {TRUTH.noise}. Twelve points leave
                these ranges wide, and the posterior shows that.
              </p>

              <p>
                Press <em>Run again</em> to use a new seed. Every draw changes, and the picture changes little.
              </p>

              <Code source={source} title="compute.ts" />
            </>
          )
        }}
      </Results>
    </StoryPage>
  )
}
