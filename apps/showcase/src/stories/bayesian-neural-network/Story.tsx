import type { ReactNode } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Aside, Code, Figure } from '../../kit/Blocks.tsx'
import { Play } from '../../kit/Play.tsx'
import { Results } from '../../kit/Results.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'
import { POINTS } from './data.ts'
import { NetworksFigure } from './Figures.tsx'
import { readResult } from './reading.ts'

const fixed = (v: number, digits = 2): string => v.toFixed(digits)

/** 'A neural network and its uncertainty': the narrative, with the play button where the computation starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  return (
    <StoryPage>
      <p>
        Here are sixteen measurements. For each one, we set a value <em>x</em> and read off a value <em>y</em>. The
        points follow a gentle wave with a little noise. They come in two clumps. Nothing was measured in the gap
        between them, or beyond the two ends.
      </p>

      <Figure caption="Sixteen points in two clumps, with a gap between them." wide>
        <NetworksFigure data={POINTS} result={undefined} />
      </Figure>

      <p>
        Fit a small neural network to these points. It has one input, <em>x</em>, and three hidden units. Each unit
        multiplies <em>x</em> by a weight, adds a bias and applies <code>tanh</code>. The output is a weighted sum of
        the three units plus a bias. That makes ten numbers in all.
      </p>

      <p>
        The usual way to fit it is to train it. We change the ten numbers until the curve passes close to the points,
        then keep that one setting. The curve fits well inside the clumps. In the gap and beyond the ends it is just as
        smooth, but the data say nothing there. One trained network shows no sign of that.
      </p>

      <h2>Many networks instead of one</h2>

      <p>
        The Bayesian version keeps every setting of the ten numbers that fits. It first states a belief about the
        weights. A weight is probably not large, so each gets a normal distribution with a spread of 1.
      </p>

      <p>
        Then it uses the data. A network whose curve passes close to the sixteen points becomes more likely. A network
        that misses them becomes less likely. The result is a distribution over networks, called the <em>posterior</em>.
        One draw from it is one complete network.
      </p>

      <p>
        We cannot write the posterior as a formula, so we draw samples from it. The sampler is NUTS, the one that drew
        the lines in <Link to="/story/bayesian-linear-regression">A line, and how sure we are</Link>. It needs the
        gradient of the log probability, which omai computes by automatic differentiation. We then run each draw over
        inputs that reach well past the data on both sides. Each draw gives one curve.
      </p>

      <Aside>
        The ten weights are hard to read. Swap two hidden units, or flip the signs of one unit's weights, and the curve
        stays the same, because <code>tanh</code> is an odd function. Every curve comes from 48 sets of weights. That is
        why the figure shows curves.
      </Aside>

      <p>Press play to draw the networks.</p>

      <Play run={run} label="Draw the networks" estimate="about five seconds" />

      <Results run={run}>
        {(result) => {
          const r = readResult(result)
          return (
            <>
              <h2>The networks</h2>

              <p>
                The sampler kept {result.draws.toLocaleString('en-GB')} networks. The figure shows {result.fan.length}{' '}
                of them as thin curves, with one picked out in green. The grey line is the average of all of them. The
                shaded band holds the middle 90% of them at each <em>x</em>.
              </p>

              <Figure
                caption="Sampled networks (thin), one of them (green), their average (grey) and the middle 90% (shaded)."
                wide
              >
                <NetworksFigure data={POINTS} result={result} />
              </Figure>

              <p>
                Each network on its own fits the points. Over the points, the band is about{' '}
                <span className="num">{fixed(r.dataWidth)}</span> wide on average. At <em>x</em> = 0, in the middle of
                the gap, it is <span className="num">{fixed(r.gapWidth)}</span> wide, which is{' '}
                <span className="num">{fixed(r.gapRatio, 1)}</span> times as much. At <em>x</em> = −3 it is{' '}
                <span className="num">{fixed(r.leftWidth)}</span> wide, and at <em>x</em> = 3 it is{' '}
                <span className="num">{fixed(r.rightWidth)}</span> wide. That is{' '}
                <span className="num">{fixed(r.leftRatio, 1)}</span> and{' '}
                <span className="num">{fixed(r.rightRatio, 1)}</span> times the width over the points.{' '}
                {r.widerAwayFromData
                  ? 'The networks agree where there are points, and spread out where there are none.'
                  : 'In this run the band is not wider everywhere away from the points. Compare the numbers with the picture.'}
              </p>

              <p>
                The model never says to be cautious away from the data. It says only that the curve should pass near the
                sixteen points and the weights should be modest. Many curves do that, and they differ where the points
                give no guidance. The caution follows from that.
              </p>

              <h2>What to take from it</h2>

              <p>
                One trained network gives a curve like the green one, and nothing more. The posterior gives the curve
                and the doubt about it. A prediction inside a clump has a band about{' '}
                <span className="num">{fixed(r.dataWidth)}</span> wide. A prediction in the middle of the gap has a band
                about <span className="num">{fixed(r.gapWidth)}</span> wide.
              </p>

              <p>
                The band is the doubt about the curve under the data. It does not include the scatter of single new
                measurements around that curve. Press <em>Run again</em> to use a new seed. Every network changes, and
                the picture changes little.
              </p>

              <Code source={source} title="compute.ts" />
            </>
          )
        }}
      </Results>
    </StoryPage>
  )
}
