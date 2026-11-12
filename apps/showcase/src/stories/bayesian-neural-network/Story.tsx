import type { ReactNode } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Aside, Code, Figure, Reveal } from '../../kit/Blocks.tsx'
import { Play } from '../../kit/Play.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'
import { POINTS } from './data.ts'
import { NetworksFigure } from './Figures.tsx'
import { readResult } from './reading.ts'

const fixed = (v: number, digits = 2): string => v.toFixed(digits)

/** 'A network that knows what it does not know': the narrative, with the play button where the computation starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  const result = run.result
  const r = result === undefined ? undefined : readResult(result)
  return (
    <StoryPage>
      <p>
        Here are sixteen measurements, in two clumps. Something was set along the bottom, call it <em>x</em>, and
        something was read off in response, call it <em>y</em>. The points sit on a gentle wave with a little noise. In
        the middle there is a gap where nobody measured anything, and on both sides the measuring stops short.
      </p>

      <Figure
        caption={
          result === undefined
            ? 'Sixteen noisy points in two clumps. The axes already reach into the gap and out beyond the ends; keep an eye on that empty space.'
            : 'The same points with sampled networks (thin curves), one of them picked out (green), their average (grey), and the range holding 90% of them at each x (shaded).'
        }
        wide
      >
        <NetworksFigure data={POINTS} result={result} />
      </Figure>

      <p>
        Fit a small neural network to these. One input, <em>x</em>; three hidden units that each take a weighted{' '}
        <em>x</em> and squash it with <code>tanh</code>; one output, a weighted sum of the three. That is ten numbers to
        choose. The usual way is to train it: nudge the ten numbers until the curve passes close to the points, stop,
        and keep that one setting.
      </p>

      <p>
        What you get is a curve, and inside the clumps it is a good one. In the gap and past the ends it is just as
        smooth and just as sure of itself, and it has nothing to be sure about. A trained network cannot tell you the
        difference. It has one answer, and it gives that one with the same straight face everywhere.
      </p>

      <h2>Keep every network the data allow</h2>

      <p>
        The Bayesian version does not stop at one setting of the ten numbers. It starts with a belief about them before
        looking at any data (each weight is probably not enormous: a normal with a spread of 1), and then lets the
        points vote. A network whose curve passes close to the sixteen points becomes more plausible; one that misses
        them becomes less. What is left is a probability distribution over networks, the <em>posterior</em>. One draw
        from it is one complete network, weights and all.
      </p>

      <p>
        Nobody can write that distribution down, so we sample it. The sampler is NUTS, the same one that drew the lines
        in <Link to="/story/bayesian-linear-regression">A line, and how sure we are</Link>; it needs the gradient of the
        log probability through the network, and omai gets that from its own automatic differentiation. Each draw is
        then pushed through a ruler of inputs running well past the data on both sides, and each draw becomes a curve.
      </p>

      <Aside>
        The ten weights themselves are not what we will look at. Swap two hidden units, or flip the signs of one unit's
        weights (tanh is odd, so the unit gives the same output), and the network computes exactly the same curve. So
        every curve has 48 different sets of weights that stand for it, and the sampler wanders among them. The weights
        are muddled; the functions are not. That is why the figure shows functions.
      </Aside>

      <Play run={run} label="Draw the networks" estimate="about five seconds" />

      {r === undefined || result === undefined ? (
        <p className="text-muted">
          Nothing has been computed yet. When you press play, the sampler runs in a background worker and the networks
          appear in the figure above.
        </p>
      ) : (
        <Reveal>
          <h2>What came back</h2>

          <p>
            The thin curves in the figure are draws from the posterior: {result.draws.toLocaleString('en-GB')} networks
            were kept, of which it shows {result.fan.length}. One of them is picked out in green. Any of them is a
            perfectly good network that fits the points, and on its own it would tell you a confident story. The grey
            line is the average of all of them, and the shaded band holds the middle 90% of them at each <em>x</em>.
          </p>

          <p>
            Over the points, the band is about <span className="num">{fixed(r.dataWidth)}</span> wide on average. At{' '}
            <em>x</em> = 0, in the middle of the gap, it is <span className="num">{fixed(r.gapWidth)}</span> wide:{' '}
            <span className="num">{fixed(r.gapRatio, 1)}</span> times as much. Out at <em>x</em> = −3 it is{' '}
            <span className="num">{fixed(r.leftWidth)}</span> and at <em>x</em> = 3 it is{' '}
            <span className="num">{fixed(r.rightWidth)}</span>, which is{' '}
            <span className="num">{fixed(r.leftRatio, 1)}</span> and{' '}
            <span className="num">{fixed(r.rightRatio, 1)}</span> times the width over the points.{' '}
            {r.widerAwayFromData
              ? 'The networks agree where there are points to agree about, and fan out where there are none.'
              : 'In this run the band is not wider everywhere away from the points; compare the numbers with the picture.'}
          </p>

          <p>
            Nothing in the model says <em>be cautious away from the data</em>. All it says is that the curve should pass
            near the sixteen points and that the weights should be modest. Many different curves do that, and they
            differ from one another exactly where the points give them no reason not to. The caution is a consequence,
            not a rule.
          </p>

          <h2>What to take from it</h2>

          <p>
            Training one network would have given you something like the green curve, and nothing else. The posterior
            gives the curve and the doubt, and the doubt is the part you act on: a prediction inside a clump comes with
            a band about <span className="num">{fixed(r.dataWidth)}</span> wide and one in the middle of the gap with a
            band about <span className="num">{fixed(r.gapWidth)}</span> wide, and you would want to know which one you
            were looking at before sending the number anywhere.
          </p>

          <p>
            One thing the band is not: it is the doubt about the <em>function</em>, the curve under the data, and it
            leaves out the scatter of single new measurements around that curve. Press <em>Run again</em>. The sampler
            gets a new seed, so every network is different, and the picture hardly moves.
          </p>
        </Reveal>
      )}

      <Code source={source} title="compute.ts" />
    </StoryPage>
  )
}
