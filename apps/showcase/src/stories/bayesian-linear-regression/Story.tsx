import type { ReactNode } from 'react'
import { Aside, Code, Figure } from '../../kit/Blocks.tsx'
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
        Here are twelve measurements. For each one, we set a value <em>x</em> and read off a value <em>y</em>. The
        points rise from left to right, but none lies exactly on a line.
      </p>

      <Figure caption="The twelve points, between x = 0.3 and x = 4.3." wide>
        <LinesFigure data={DATA} result={undefined} />
      </Figure>

      <p>
        The usual step is to fit one line: the intercept and slope that pass closest to the points. One line does not
        say how far to trust it. Many other lines fit almost as well.
      </p>

      <h2>Many lines instead of one</h2>

      <p>
        Bayesian regression keeps every line that fits. It first states what we believe before we see the data. The
        intercept and the slope each get a normal distribution centred on 0 with a spread of 5. The noise is how far the
        points scatter around the line. It cannot be negative, so it gets a half-normal distribution with a spread of 2.
      </p>

      <p>
        Then it uses the data. A line that passes close to many points becomes more likely. A line that misses them
        becomes less likely. The result is a distribution over the intercept, slope and noise, called the{' '}
        <em>posterior</em>. One draw from it is one line.
      </p>

      <p>
        We cannot write the posterior as a formula, so we draw samples from it. The sampler is NUTS. It takes steps
        through the intercept, slope and noise, guided by the gradient of the log probability. Each point it visits is
        one line.
      </p>

      <Aside>
        NUTS needs a gradient. omai computes it by automatic differentiation, so the model does not supply one.
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
                The sampler kept {result.draws.toLocaleString('en-GB')} draws. Each draw is one line, and the figure
                shows {result.fan.length} of them. The solid line is their average. The shaded band holds the middle 90%
                of the lines at each <em>x</em>.
              </p>

              <Figure
                caption="Sampled lines (thin), their average (solid), the middle 90% (shaded) and the line the data came from (dashed)."
                wide
              >
                <LinesFigure data={DATA} result={result} />
              </Figure>

              <p>
                Where the lines are close together, the data fix them. Near <em>x</em> ={' '}
                <span className="num">{fixed(r.centre, 1)}</span>, in the middle of the points, the band is{' '}
                <span className="num">{fixed(r.centreWidth)}</span> wide. At <em>x</em> ={' '}
                <span className="num">{fixed(r.edge, 0)}</span>, far to the right of the last point, it is{' '}
                <span className="num">{fixed(r.edgeWidth)}</span> wide, about{' '}
                <span className="num">{fixed(r.ratio, 1)}</span> times as wide. There are no measurements out there, so
                the lines disagree.
              </p>

              <p>The next figure plots each line as one dot, by its intercept and slope.</p>

              <Figure caption="Each dot is one sampled line. The marked dot is the line the data came from.">
                <CloudFigure result={result} />
              </Figure>

              <p>
                The dots form a tilted cloud. A steeper line needs a lower intercept to stay near the points, so the two
                move in opposite directions. This is why the lines pivot about the middle of the data. There, a change
                of slope moves a line a little. Far from the middle, it moves a line a lot.
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
                <span className="num">{fixed(r.noise.hi)}</span>. The true value is {TRUTH.noise}. With only twelve
                points these ranges are wide, and the posterior shows that.
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
