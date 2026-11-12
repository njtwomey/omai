import type { ReactNode } from 'react'
import { Aside, Code, Figure } from '../../kit/Blocks.tsx'
import { Play } from '../../kit/Play.tsx'
import { Results } from '../../kit/Results.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'
import { simulate } from './data.ts'
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
        Here are twelve measurements. For each one, we set a value <em>x</em> and read off a value <em>y</em>. The
        points rise from left to right. None of them lies exactly on a line.
      </p>

      <Figure caption="The twelve points. They lie between x = 0.3 and x = 4.3." wide>
        <LinesFigure data={DATA} result={undefined} />
      </Figure>

      <p>
        The usual first step is to fit a line: the intercept and the slope that pass closest to the points. That gives
        one line. It does not say how far to trust it. Twelve noisy points fit many lines: some a little steeper, some
        flatter, some higher or lower.
      </p>

      <h2>Many lines instead of one</h2>

      <p>
        Bayesian regression keeps every line that fits. It starts from what we believe before we look at the data. Here
        the intercept and the slope each get a normal distribution centred on 0 with a spread of 5. The noise is how far
        the points scatter around the line. It must be positive, so it gets a half-normal distribution with a spread of
        2.
      </p>

      <p>
        Then it uses the data. A line that passes close to many points becomes more likely. A line that misses them
        becomes less likely. The result is a probability distribution over the intercept, the slope and the noise. It is
        called the <em>posterior</em>. One draw from the posterior is one line.
      </p>

      <p>
        We cannot write the posterior as a formula, so we draw samples from it. The sampler is NUTS, a Hamiltonian Monte
        Carlo method. It moves through the space of intercept, slope and noise in steps guided by the gradient of the
        log probability, and it keeps the points it visits. Each kept point is one line.
      </p>

      <Aside>
        NUTS needs the gradient of the log probability. The model is not told how to compute it. omai gets the gradient
        by automatic differentiation.
      </Aside>

      <p>Press play to draw the lines.</p>

      <Play run={run} label="Draw the lines" estimate="a few seconds" />

      <Results run={run}>
        {(result) => {
          const r = readResult(result, DATA)
          return (
            <>
              <h2>The lines</h2>

              <p>
                The sampler kept {result.draws.toLocaleString('en-GB')} draws. Each draw is one line. The figure shows{' '}
                {result.fan.length} of them. The solid line is their average. The shaded band holds the middle 90% of
                the lines at each <em>x</em>.
              </p>

              <Figure
                caption="The twelve points with sampled lines (thin), their average (solid), the range holding 90% of the lines at each x (shaded) and the line the data were drawn from (dashed)."
                wide
              >
                <LinesFigure data={DATA} result={result} />
              </Figure>

              <p>
                Where the lines are close together, the data fix them. Where they spread out, the data do not. Near{' '}
                <em>x</em> = <span className="num">{fixed(r.centre, 1)}</span>, in the middle of the points, the band is{' '}
                <span className="num">{fixed(r.centreWidth)}</span> wide. At <em>x</em> ={' '}
                <span className="num">{fixed(r.edge, 0)}</span>, far to the right of the last point, it is{' '}
                <span className="num">{fixed(r.edgeWidth)}</span> wide, about{' '}
                <span className="num">{fixed(r.ratio, 1)}</span> times as wide. There are no measurements out there. The
                lines agree where the data are and disagree where they are not.
              </p>

              <p>
                The next figure shows the same draws as points. Each dot is one line, placed by its intercept and its
                slope.
              </p>

              <Figure caption="Each dot is one sampled line, placed by its intercept and slope. The marked dot is the line the data were drawn from.">
                <CloudFigure result={result} />
              </Figure>

              <p>
                The dots form a tilted cloud. A steeper line needs a lower intercept to stay near the points, so the two
                values move in opposite directions. This is why the lines in the first figure pivot about the middle of
                the data. Near the middle, a change of slope moves the line a little. Far from the middle, it moves the
                line a lot.
              </p>

              <h2>What to take from it</h2>

              <p>
                A single fit gives one value for the slope. The posterior also gives the doubt. The slope has a mean of{' '}
                <span className="num">{fixed(r.slope.mean)}</span>, and nine in ten of the draws lie between{' '}
                <span className="num">{fixed(r.slope.lo)}</span> and <span className="num">{fixed(r.slope.hi)}</span>.
                The line the data were drawn from has a slope of 0.7, which is {r.slopeInside ? 'inside' : 'outside'}{' '}
                that range. The noise has a mean of <span className="num">{fixed(r.noise.mean)}</span>, and the true
                value is 0.9. Nine in ten of its draws lie between <span className="num">{fixed(r.noise.lo)}</span> and{' '}
                <span className="num">{fixed(r.noise.hi)}</span>. With twelve points these ranges are wide, and the
                posterior shows that.
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
