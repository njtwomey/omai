import type { ReactNode } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Aside, Code, Figure } from '../../kit/Blocks.tsx'
import { Equation, Tex, tex } from '../../kit/Maths.tsx'
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
        To fit a curve to data, we usually choose a form first: a line, a polynomial or a spline. The data then set the
        few numbers inside that form.
      </p>

      <p>
        A Gaussian process does not choose one curve. It puts a probability on every smooth function. The data then
        remove the functions that do not fit. What is left is every function that is still possible. Where those
        functions disagree, the data say little.
      </p>

      <h2>A probability on whole functions</h2>

      <p>
        A function has a value at every <Tex>x</Tex>, which is too many numbers to give a probability. So we look at it
        in a finite number of places: here a grid of 121 points from <Tex>x = -3</Tex> to <Tex>x = 3</Tex>. A Gaussian
        process says the values at these places are jointly normal. Two things set that distribution: the average value,
        zero here, and how strongly the value at one <Tex>x</Tex> goes with the value at another.
      </p>

      <Equation label="A Gaussian process" reading="The function f is a Gaussian process with mean zero and kernel k.">
        {String.raw`f \sim \GP\big(0,\; k(x, x')\big)`}
      </Equation>

      <p>
        The second part is the <em>kernel</em>. This story uses the squared exponential kernel. Values at nearby{' '}
        <Tex>x</Tex> move together almost exactly, and values far apart hardly move together. The length scale{' '}
        <Tex>\ell</Tex>, here <span className="num">{fixed(LENGTH_SCALE)}</span>, is the distance over which this link
        fades. The functions are therefore smooth, and they bend on a scale of about that length.
      </p>

      <Equation
        label={
          <>
            The squared exponential kernel, with length scale <Tex>\ell</Tex>
          </>
        }
        reading="The kernel of x and x prime is the exponential of minus the squared distance between x and x prime, divided by two times the length scale squared."
      >
        {String.raw`k(x, x') = \exp\!\left(-\frac{(x - x')^2}{2\,\ell^2}\right)`}
      </Equation>

      <Aside>
        A Gaussian process needs no sampler. Drawing a function and conditioning on data are both linear algebra on a
        covariance matrix, using its Cholesky factorisation.
      </Aside>

      <p>
        When you press play, eight functions are drawn from this prior, before there is any data. Then five points are
        made from one more function of the same process, with noise of standard deviation{' '}
        <span className="num">{fixed(NOISE, 1)}</span>. That function stays hidden. The process is conditioned on the
        five points in one step, and eight functions are drawn again.
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
                caption="Eight functions from the prior (thin) and the band holding 90% of all such functions (shaded)."
                wide
              >
                <FunctionsFigure grid={result.grid} crowd={result.prior} />
              </Figure>

              <p>
                The eight functions are all different, and all smooth. None is preferred. The shaded band is{' '}
                <span className="num">{fixed(r.priorWidth)}</span> wide at every <Tex>x</Tex>, because the process has
                been told nothing that would make it surer about one place than another.
              </p>

              <h2>After five points</h2>

              <p>Now the process is conditioned on the five points, and eight functions are drawn again.</p>

              <Figure
                caption="Eight functions from the process conditioned on the five points (thin), their mean (thick) and the band holding 90% of them (shaded)."
                wide
              >
                <FunctionsFigure grid={result.grid} crowd={result.posterior} showMean data={result.data} />
              </Figure>

              <p>
                Every one of the eight functions now passes near the points. The largest distance between any of them
                and a reading, at the nearest place on the grid, is <span className="num">{fixed(r.largestMiss)}</span>.
                For the eight functions of the prior, it is <span className="num">{fixed(r.priorMiss)}</span>.
              </p>

              <p>
                Away from the points, the functions can differ, and they do. At the five points the band is{' '}
                <span className="num">{fixed(r.atPoints)}</span> wide on average. Between them it widens. At{' '}
                <Tex>{tex`x = ${fixed(r.betweenAt, 1)}`}</Tex>, the widest place between the outer points, it is{' '}
                <span className="num">{fixed(r.between)}</span> wide, and the eight functions there range from{' '}
                <span className="num">{fixed(r.betweenLow)}</span> to{' '}
                <span className="num">{fixed(r.betweenHigh)}</span>.{' '}
                {r.widest > r.between + 0.05 ? (
                  <>
                    The band is widest of all at <Tex>{tex`x = ${fixed(r.widestAt, 1)}`}</Tex>, beyond the outer points,
                    where it is <span className="num">{fixed(r.widest)}</span> wide. For the prior it is{' '}
                    <span className="num">{fixed(r.priorWidth)}</span> wide.
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
                The thick line is the posterior mean, the best single guess. The band is not an error bar added
                afterwards. It comes from the same calculation as the mean, because conditioning on data changes the
                whole distribution of functions, not just its middle.
              </p>

              <p>
                The only choices were the kernel and its length scale. A smaller length scale gives rougher functions,
                and a band that closes only very near the points. In{' '}
                <Link to={storyHref('bayesian-linear-regression')}>A line, and how sure we are</Link> the crowd was
                lines, which have only an intercept and a slope to be unsure about. Here the crowd is whole curves, and
                the only requirement is that they are smooth.
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
