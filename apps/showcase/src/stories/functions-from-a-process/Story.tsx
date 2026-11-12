import type { ReactNode } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Aside, Code, Figure } from '../../kit/Blocks.tsx'
import { Play } from '../../kit/Play.tsx'
import { Results } from '../../kit/Results.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import { storyHref } from '../../story.ts'
import { LENGTH_SCALE, NOISE, type Result } from './compute.ts'
import source from './compute.ts?raw'
import { FunctionsFigure } from './Figures.tsx'
import { readResult } from './reading.ts'

const fixed = (v: number, digits = 2): string => v.toFixed(digits)

/** 'Functions drawn from a Gaussian process': the narrative, with the play button where the computation starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  return (
    <StoryPage>
      <p>
        Fitting a curve to data means choosing a form first: a line, a polynomial of some degree, or a spline with some
        number of knots. The data then set the few numbers inside that form. The form is chosen before the first point
        arrives.
      </p>

      <p>
        A Gaussian process works differently. It does not choose one curve. It puts a probability on every smooth
        function. The data are then used to remove the functions that do not fit. What is left is every function that is
        still possible. Where those functions disagree, the data say little.
      </p>

      <h2>A probability on whole functions</h2>

      <p>
        A function has a value at every <em>x</em>. That is too many numbers to put a probability on. In practice we
        look at a function in a finite number of places. Here the places are a grid of 121 points from <em>x</em> = −3
        to <em>x</em> = 3. A Gaussian process says that the values at any such set of places are jointly normal. Two
        things set this distribution: the average value, which is zero here, and how strongly the value at one{' '}
        <em>x</em> goes with the value at another.
      </p>

      <p>
        The second part is called the <em>kernel</em>. This story uses the squared exponential kernel. Values at nearby{' '}
        <em>x</em> move together almost exactly. Values far apart hardly move together. The length scale, here{' '}
        <span className="num">{fixed(LENGTH_SCALE)}</span>, is the distance over which the link fades. So the functions
        are smooth, and they bend on a scale of about half a unit of <em>x</em>. The kernel says nothing about where the
        functions go.
      </p>

      <Aside>
        A Gaussian process needs no sampler. Drawing a function and conditioning on data are both linear algebra on a
        covariance matrix, and both use its Cholesky factorisation.
      </Aside>

      <p>
        When you press play, the first step is to draw eight functions from this prior, before there is any data. Next,
        five points are made by reading one more function from the same process, with noise of standard deviation{' '}
        <span className="num">{fixed(NOISE, 1)}</span>. That function stays hidden. Then the process is conditioned on
        the five points in one step, and eight functions are drawn again.
      </p>

      <Play run={run} label="Draw the functions" estimate="under a second" />

      <Results run={run}>
        {(result) => {
          const r = readResult(result)
          return (
            <>
              <h2>Before any data</h2>

              <p>These are the eight functions drawn from the prior.</p>

              <Figure
                caption="Eight functions drawn from the prior (thin lines) and the band holding 90% of all such functions at each x (shaded)."
                wide
              >
                <FunctionsFigure grid={result.grid} crowd={result.prior} />
              </Figure>

              <p>
                The eight functions are all different, and all smooth. They swing up and down and rarely go more than a
                couple of units from zero. None is preferred. The shaded band is{' '}
                <span className="num">{fixed(r.priorWidth)}</span> wide at every <em>x</em>, because the process has not
                been told anything that would make it surer about one place than another.
              </p>

              <h2>After five points</h2>

              <p>Now the process is conditioned on the five points, and eight functions are drawn again.</p>

              <Figure
                caption="The same process conditioned on the five points. Eight new functions (thin lines), their mean (thick) and the band holding 90% of them at each x (shaded)."
                wide
              >
                <FunctionsFigure grid={result.grid} crowd={result.posterior} showMean data={result.data} />
              </Figure>

              <p>
                Every one of the eight functions now passes near the points. The largest distance between any of them
                and a reading, at the nearest place on the grid, is <span className="num">{fixed(r.largestMiss)}</span>.
                For the eight functions of the prior, the same distance is{' '}
                <span className="num">{fixed(r.priorMiss)}</span>.
              </p>

              <p>
                Away from the points, the functions can differ, and they do. At the five points the band is{' '}
                <span className="num">{fixed(r.atPoints)}</span> wide on average. Between them it widens. At <em>x</em>{' '}
                = <span className="num">{fixed(r.betweenAt, 1)}</span>, the widest place between the outer points, it is{' '}
                <span className="num">{fixed(r.between)}</span> wide, and the eight functions there range from{' '}
                <span className="num">{fixed(r.betweenLow)}</span> to{' '}
                <span className="num">{fixed(r.betweenHigh)}</span>.{' '}
                {r.widest > r.between + 0.05 ? (
                  <>
                    The band is widest of all at <em>x</em> = <span className="num">{fixed(r.widestAt, 1)}</span>,
                    beyond the outer points, where it is <span className="num">{fixed(r.widest)}</span> wide. For the
                    prior it is <span className="num">{fixed(r.priorWidth)}</span> wide.
                  </>
                ) : (
                  <>
                    Beyond the outer points the band cannot be wider than the prior&rsquo;s{' '}
                    <span className="num">{fixed(r.priorWidth)}</span>, which it approaches as the data fall away.
                  </>
                )}
              </p>

              <h2>What to take from it</h2>

              <p>
                The thick line is the mean of the posterior, the best single guess. It is also the average of many
                functions that disagree in places. The band is not an error bar added afterwards. It comes from the same
                calculation as the mean, because conditioning on data changes the whole distribution of functions, not
                just its middle.
              </p>

              <p>
                The only choices were the kernel and its length scale. They decide how the functions look before and
                after the data. A smaller length scale gives rougher functions, and a band that closes only very near
                the points. In <Link to={storyHref('bayesian-linear-regression')}>A line, and how sure we are</Link> the
                crowd was lines, which have only an intercept and a slope to be unsure about. Here the crowd is whole
                curves, and the only requirement is that they are smooth.
              </p>

              <p>
                Press <em>Run again</em> to use a new seed. The functions, the five points and their places all change.
                The band still closes at the new points and opens between them.
              </p>

              <Code source={source} title="compute.ts" />
            </>
          )
        }}
      </Results>
    </StoryPage>
  )
}
