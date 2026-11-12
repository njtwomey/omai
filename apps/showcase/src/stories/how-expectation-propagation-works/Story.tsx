import { type ModelRun, modelRunAt } from 'omai-engine/exposition'
import { messageTex } from 'omai-render/views'
import { Fragment, type ReactNode, useCallback } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Code, Figure } from '../../kit/Blocks.tsx'
import { Equation, Tex, tex } from '../../kit/Maths.tsx'
import { MessageReplay } from '../../kit/MessageReplay.tsx'
import { Play } from '../../kit/Play.tsx'
import { Results } from '../../kit/Results.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import { storyHref } from '../../story.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'
import { CLUTTER_CHANCE, CLUTTER_SD, MAX_SWEEPS, PRIOR_SD, ROWS, SHOWN_SWEEP, TOLERANCE } from './data.ts'
import { AnswerFigure, ChangeFigure, GraphArt, SweepsFigure, UpdateFigure } from './Figures.tsx'
import {
  bellText,
  clutterPositions,
  fixed,
  listOf,
  messageText,
  readExact,
  readSweeps,
  sci,
  weakOnes,
} from './reading.ts'

const FIRST_SEED = 37

const linkClass = 'underline underline-offset-2'

// The worked example of the natural parameters: the prior's precision, and the product of the prior and a bell with mean 2 and sd 1.
const PRIOR_PRECISION = 1 / PRIOR_SD ** 2
const PRODUCT = {
  precision: 1 + PRIOR_PRECISION,
  mean: 2 / (1 + PRIOR_PRECISION),
  sd: 1 / Math.sqrt(1 + PRIOR_PRECISION),
}

/** What happens at one step of the replay, in words: which factor was updated, what it sent, and the belief after it. */
function describeUpdate(step: number, record: ModelRun, points: readonly number[]): ReactNode {
  const at = modelRunAt(record, step)
  const sent = at?.messages.at(-1)
  if (at === undefined || sent === undefined)
    return (
      <>
        <p className="m-0 font-medium">Before the first update</p>
        <p className="m-0">Every stand-in is flat, so the unknown has no belief yet.</p>
      </>
    )
  const factor = Number(sent.from.replace(/^\D+/, ''))
  const sweep = Math.ceil(step / (points.length + 1))
  const who =
    factor === 0 ? (
      'the prior'
    ) : (
      <>
        measurement {factor} (<Tex>{tex`y = ${fixed(points[factor - 1] as number)}`}</Tex>)
      </>
    )
  const belief = Object.values(at.nodes).find((n) => n.mean !== undefined && n.sd !== undefined)
  const sentTex = messageTex(sent.text)
  return (
    <>
      <p className="m-0 font-medium">
        Update {step}, in sweep {sweep}: {who}
      </p>
      <p className="m-0">It sends {sentTex === undefined ? sent.text : <Tex>{sentTex}</Tex>}.</p>
      <p className="m-0">
        {belief?.mean === undefined || belief.sd === undefined
          ? 'The unknown has no belief yet.'
          : `The belief about the location is now mean ${fixed(belief.mean)}, standard deviation ${fixed(belief.sd)}.`}
      </p>
    </>
  )
}

/** A measurement as a row label: `y_3 = 0.84`, the value read from the run. */
const yIs = (index: number, value: number): ReactNode => <Tex>{`y_{${index}} = ${fixed(value)}`}</Tex>

/** Pieces of maths or words as a list in words: `a, b and c`. */
const listNodes = (items: readonly ReactNode[]): ReactNode =>
  items.map((item, i) => (
    // biome-ignore lint/suspicious/noArrayIndexKey: a fixed list, never reordered
    <Fragment key={i}>
      {i === 0 ? '' : i === items.length - 1 ? ' and ' : ', '}
      {item}
    </Fragment>
  ))

/** A small table in the page's sans-serif, scrolling sideways on a narrow screen. */
function Table(props: {
  readonly head: readonly ReactNode[]
  readonly rows: readonly (readonly ReactNode[])[]
}): ReactNode {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse font-sans text-[0.9rem] tabular-nums">
        <thead>
          <tr>
            {props.head.map((h, i) => (
              <th
                // biome-ignore lint/suspicious/noArrayIndexKey: the columns are a fixed list, never reordered
                key={i}
                className="whitespace-nowrap border-b border-line px-2.5 py-1.5 text-left font-semibold text-muted"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {props.rows.map((row, i) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: the rows are a fixed list, never reordered
            <tr key={i} className="border-b border-line last:border-0">
              {row.map((cell, j) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: the cells are a fixed list, never reordered
                <td key={j} className="whitespace-nowrap px-2.5 py-1.5 align-top">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** 'How expectation propagation works': the narrative, with the play button where the first computed figure is needed. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(
    () => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }),
    FIRST_SEED,
  )
  return (
    <StoryPage>
      <p>
        Expectation propagation, or EP, approximates the posterior of a model when the posterior cannot be written down.
        This post follows one run of omai&rsquo;s EP on one small model, in the order things happen. After the play
        button, every number comes from that run.
      </p>

      <p>
        At the end you should be able to say what EP does to each factor of a model. You should also be able to read an
        EP run in omai: its messages, its sweeps and the point where it stops.
      </p>

      <h2>1. The problem</h2>

      <p>
        We start with a model and some data. The model says how the data could have come from some unknown numbers. We
        want the posterior: what we believe about the unknowns once we have seen the data. By Bayes&rsquo; rule, the
        posterior is the prior (what we believed before) times the likelihood of each data point, divided by one
        constant that makes the total probability 1.
      </p>

      <Equation
        label={
          <>
            Bayes&rsquo; rule, with <Tex>Z</Tex> the constant
          </>
        }
        reading="The posterior of theta given the data y equals one over Z, times the prior of theta, times the product over the data points of the likelihood of each point."
      >
        {String.raw`p(\theta \mid \yvec) = \frac{1}{Z}\, p(\theta) \prod_{i} p(y_i \mid \theta)`}
      </Equation>

      <p>
        That product is the difficulty. For a few models it has a known shape, and the posterior is a formula. For most
        it does not. The factors have different shapes, their product has no name, and the constant is an integral with
        no closed form. With one unknown we can still compute the posterior numerically, and we will, to check the
        result. With thousands of unknowns we cannot.
      </p>

      <p>
        So we settle for an approximation we can handle. EP uses the Gaussian, or bell curve. A bell has two numbers:
        its mean <Tex>\mu</Tex>, where it is centred, and its variance <Tex>\sigma^2</Tex>, how wide it is. The standard
        deviation, written sd or <Tex>\sigma</Tex>, is the square root of the variance. EP&rsquo;s answer is one bell
        for each unknown, and its mean and spread are the belief.
      </p>

      <Equation
        label="A bell with mean μ and variance σ²"
        reading="A bell curve in theta with mean mu and variance sigma squared is proportional to the exponential of minus theta minus mu, squared, over two sigma squared."
      >
        {String.raw`\Gauss(\theta;\, \mu, \sigma^2) \;\propto\; \exp\!\left(-\frac{(\theta - \mu)^2}{2\sigma^2}\right)`}
      </Equation>

      <p>
        The model for the rest of the post is Minka&rsquo;s clutter problem, from the paper that introduced EP (2001).
        There is one unknown, a location, written <Tex>\theta</Tex>. We have {ROWS} measurements. With probability{' '}
        {fixed(1 - CLUTTER_CHANCE, 1)}, a measurement is a reading of the location: a draw from a bell with sd 1 around{' '}
        <Tex>\theta</Tex>. With probability {fixed(CLUTTER_CHANCE, 1)} it is clutter: a draw from a much wider bell,
        with sd {fixed(CLUTTER_SD, 1)}, centred on zero and unrelated to <Tex>\theta</Tex>.
      </p>

      <p>
        We are not told which measurements are clutter. Before seeing them we believe <Tex>\theta</Tex> is within about{' '}
        {PRIOR_SD} of zero: a bell with mean 0 and sd {PRIOR_SD}.
      </p>

      <Equation
        label="The posterior of the clutter model"
        reading={`The posterior of theta is proportional to a bell with mean 0 and sd ${PRIOR_SD}, times, for each of the ${ROWS} measurements, ${fixed(1 - CLUTTER_CHANCE, 1)} times a bell around theta with sd 1, plus ${fixed(CLUTTER_CHANCE, 1)} times a bell around 0 with sd ${fixed(CLUTTER_SD, 1)}.`}
      >
        {String.raw`\begin{aligned}
          p(\theta \mid \yvec) \;\propto\;{}& \Gauss(\theta;\, 0,\, ${PRIOR_SD}^2) \\
          &\times \prod_{i=1}^{${ROWS}} \Big[ ${fixed(1 - CLUTTER_CHANCE, 1)}\, \Gauss(y_i;\, \theta,\, 1^2) \\
          &\hphantom{\times \prod_{i=1}^{${ROWS}} \Big[} + ${fixed(CLUTTER_CHANCE, 1)}\, \Gauss(y_i;\, 0,\, ${fixed(CLUTTER_SD, 1)}^2) \Big]
        \end{aligned}`}
      </Equation>

      <p>
        Each of the {ROWS} brackets is a sum of two bells. Multiplying them out gives up to{' '}
        <Tex>{`2^{${ROWS}} = ${2 ** ROWS}`}</Tex> bell-shaped terms, one for each way of deciding which measurements are
        clutter. With one unknown we can evaluate the whole product on a fine grid of values of <Tex>\theta</Tex> and
        get the exact posterior. We do that at the end. EP does not use a grid, and its work grows with the number of
        factors, not with the {2 ** ROWS}.
      </p>

      <h2>2. The model as a graph of factors</h2>

      <p>
        Each part of the formula is a factor: a function of the unknowns that the posterior is multiplied by. There is
        one for the prior and one for each measurement, so {ROWS + 1} in all. A factor graph draws them: a circle for
        each unknown, a square for each factor, and a line from each factor to the unknowns it mentions.
      </p>

      <Figure
        caption={
          <>
            The factor graph of the model: one unknown, the prior factor <Tex>f_0</Tex> and the measurement factors{' '}
            <Tex>f_1</Tex> to <Tex>{`f_{${ROWS}}`}</Tex>.
          </>
        }
      >
        <GraphArt />
      </Figure>

      <p>
        Here there is one unknown, so every factor joins the same circle. In a larger model a factor joins only the few
        unknowns it mentions, and each unknown hears only from the factors around it. That keeps EP cheap on big models.
        One unknown shows every step of the method except passing messages between different unknowns.
      </p>

      <h2>3. The idea: a stand-in for each factor</h2>

      <p>
        The product of the factors is hard because the factors are not bells. EP replaces each factor by a bell, called
        a stand-in, so the product of the stand-ins is easy. A product of bells is a bell, so the approximation of the
        posterior is the product of all the stand-ins. Call it <Tex>q</Tex>.
      </p>

      <p>
        A stand-in sent from a factor to an unknown is called a message. A message is a bell, up to a constant that is
        never tracked. A bell can be written in two ways. One is the mean and the variance. The engine uses the other:
        the precision <Tex>J</Tex>, which is one over the variance, and <Tex>h</Tex>, the precision times the mean.
        Together they are the natural parameters of the bell.
      </p>

      <Equation
        label="The natural parameters of a bell"
        reading="J equals one over sigma squared, and h equals J times mu."
      >
        {String.raw`J = \frac{1}{\sigma^2}, \qquad h = J\,\mu`}
      </Equation>

      <p>
        Natural parameters turn multiplying two bells into adding. If two messages have <Tex>(h_1, J_1)</Tex> and{' '}
        <Tex>(h_2, J_2)</Tex>, their product has <Tex>(h_1 + h_2,\; J_1 + J_2)</Tex>. Dividing one bell by another means
        subtracting. A message has mean <Tex>h / J</Tex> and variance <Tex>1 / J</Tex>.
      </p>

      <p>
        For example, the prior has mean 0 and sd {PRIOR_SD}, so <Tex>{`J = ${PRIOR_PRECISION}`}</Tex> and{' '}
        <Tex>h = 0</Tex>. A bell with mean 2 and sd 1 has <Tex>J = 1</Tex> and <Tex>h = 2</Tex>. Their product has{' '}
        <Tex>{`J = ${fixed(PRODUCT.precision)}`}</Tex> and <Tex>h = 2</Tex>, which is a bell with mean{' '}
        {fixed(PRODUCT.mean)} and sd {fixed(PRODUCT.sd, 3)}.
      </p>

      <p>
        Two more facts. A message that says nothing has <Tex>J = 0</Tex> and <Tex>h = 0</Tex>. We call it flat. A
        message may also have a negative <Tex>J</Tex>. It is then not a bell, only a pair of numbers that still add and
        subtract correctly. That is allowed as long as the sum of all the messages, the approximation <Tex>q</Tex>{' '}
        itself, has a positive <Tex>J</Tex>.
      </p>

      <p>
        EP starts with every stand-in flat, except the prior&rsquo;s, which is already a bell and is used as it is. The
        task is to choose the other {ROWS} stand-ins. A good stand-in need not match its factor everywhere. It has to
        match where the posterior has its mass, and that depends on the other factors.
      </p>

      <p>
        So EP cannot choose each stand-in alone. It chooses one at a time, with the others held fixed, and repeats. One
        such choice is called an update.
      </p>

      <h2>4. One update, step by step</h2>

      <p>
        An update has five steps. We write them for a factor <Tex>k</Tex> with real factor <Tex>f_k</Tex> and current
        stand-in <Tex>g_k</Tex>. The current approximation <Tex>q</Tex> is the product of all the stand-ins.
      </p>

      <Equation
        label="The approximation is the product of the stand-ins"
        reading="q of theta equals the product over all factors k of g k of theta."
      >
        {String.raw`q(\theta) = \prod_{k} g_k(\theta)`}
      </Equation>

      <ol className="list-decimal space-y-4 pl-6">
        <li>
          Take the factor&rsquo;s own stand-in out. Divide <Tex>q</Tex> by <Tex>g_k</Tex>. The result is the cavity:
          what we believe about <Tex>\theta</Tex> from everything except factor <Tex>k</Tex>. In natural parameters,
          subtract the message.
          <Equation reading="The cavity of theta is q of theta divided by g k of theta. In natural parameters, h and J are h q minus h k, and J q minus J k.">
            {String.raw`\begin{aligned}
              \text{cavity}(\theta) &= q(\theta) / g_k(\theta) \\
              (h, J) &= (h_q - h_k,\; J_q - J_k)
            \end{aligned}`}
          </Equation>
        </li>
        <li>
          Multiply the cavity by the real factor, not its stand-in. The result is the tilted distribution: what we would
          believe if factor <Tex>k</Tex> were exact and the others still approximate. It is usually not a bell, because{' '}
          <Tex>f_k</Tex> is not one.
          <Equation reading="The tilted distribution of theta is proportional to the cavity of theta times f k of theta.">
            {String.raw`\text{tilted}(\theta) \propto \text{cavity}(\theta)\, f_k(\theta)`}
          </Equation>
        </li>
        <li>
          Find the bell with the same mean and variance as the tilted distribution. This is moment matching. Of all
          bells, it is the one with the smallest Kullback&ndash;Leibler divergence from the tilted distribution to the
          bell.
          <Equation reading="The matched bell is the bell q prime that makes the Kullback-Leibler divergence from the tilted distribution to q prime smallest.">
            {String.raw`\text{matched} = \argmin_{q' \in \text{bells}} \KL\!\left(\text{tilted} \,\middle\|\, q'\right)`}
          </Equation>
          It needs two integrals, the mean and the variance of the tilted distribution. This is the only step that uses
          the real factor, and the only one that can be hard. The library has closed forms for the factors that allow
          them and numerical integration for the rest. The clutter factor here uses numerical integration.
          <Equation reading="The mean m is the integral of theta times the tilted distribution. The variance v is the integral of theta minus m, squared, times the tilted distribution. J is one over v, and h is m over v.">
            {String.raw`\begin{aligned}
              m &= \int \theta\, \text{tilted}(\theta) \dd\theta \\
              v &= \int (\theta - m)^2\, \text{tilted}(\theta) \dd\theta \\
              J &= 1 / v, \qquad h = m / v
            \end{aligned}`}
          </Equation>
        </li>
        <li>
          Divide the matched bell by the cavity. The result is the new stand-in. In natural parameters, subtract the
          cavity from the matched bell.
          <Equation reading="The new g k of theta is the matched bell divided by the cavity. In natural parameters, h and J are h matched minus h cavity, and J matched minus J cavity.">
            {String.raw`\begin{aligned}
              g_k^{\text{new}}(\theta) &= \text{matched}(\theta) / \text{cavity}(\theta) \\
              (h, J) &= (h_{\text{matched}} - h_{\text{cavity}},\; J_{\text{matched}} - J_{\text{cavity}})
            \end{aligned}`}
          </Equation>
        </li>
        <li>
          Put the new stand-in in place of the old one. The approximation is now the cavity times the new stand-in,
          which is the matched bell. After the update, <Tex>q</Tex> has the same mean and variance as the tilted
          distribution of factor <Tex>k</Tex>.
        </li>
      </ol>

      <p style={{ marginTop: '1.25em' }}>
        To see these steps with real numbers we need a run. Press play once. In a background worker, the page draws{' '}
        {ROWS} measurements from the seed and runs omai&rsquo;s EP on the model above, one factor at a time, until it
        settles. It then takes one update apart with tensor operations and compares it with the message the engine sent.
        Everything below is read from that one run.
      </p>

      <Play run={run} label="Run expectation propagation" estimate="about a second" />

      <Results run={run} placeholder="Press play. The rest of the post appears here, with the numbers of one run.">
        {(result) => (
          <>
            <RunStory result={result} />
            <Code source={source} title="compute.ts" />
          </>
        )}
      </Results>
    </StoryPage>
  )
}

/** Everything that reads the result, from the first update to the closing pointers. */
export function RunStory(props: { readonly result: Result }): ReactNode {
  const { result } = props
  // the replay makes its words again when this function changes, so it is made once per result
  const describe = useCallback(
    (step: number, record: ModelRun): ReactNode => describeUpdate(step, record, result.points),
    [result.points],
  )
  const taken = result.taken
  const sweeps = readSweeps(result)
  const exact = readExact(result)
  const clutter = clutterPositions(result)
  const weak = weakOnes(result)
  const weakAreClutter = weak.every((j) => result.clutter[j - 1] === true)
  const cavityMean = taken === undefined ? Number.NaN : taken.cavity.shift / taken.cavity.precision
  const cavitySd = taken === undefined ? Number.NaN : 1 / Math.sqrt(taken.cavity.precision)
  const beforeBell =
    taken === undefined
      ? null
      : { mean: taken.before.shift / taken.before.precision, sd: 1 / Math.sqrt(taken.before.precision) }
  const sweepTable = result.standIns
  const beliefAfter = (s: number): string =>
    bellText({ mean: result.beliefs.mean[s - 1] as number, sd: result.beliefs.sd[s - 1] as number })
  const firstUpdates = result.first
  const second = firstUpdates[1]
  const lastUpdate = firstUpdates.at(-1)
  return (
    <>
      <h2>4, continued: one update with numbers</h2>

      <p>
        The run drew {result.rows} measurements: {listOf(result.points.map((p) => fixed(p, 2)))}. The generator also
        kept which of them were clutter
        {clutter.length === 0 ? (
          <>: none were, this time.</>
        ) : (
          <>
            : {clutter.length === 1 ? 'measurement' : 'measurements'} {listOf(clutter)}.
          </>
        )}{' '}
        EP is not told this. It sees only the values.
      </p>

      {taken === undefined ? (
        <p>
          In this run no factor had a usable cavity in sweep {SHOWN_SWEEP}, so there is no update to take apart here.
          Press <em>Run again</em> for other measurements.
        </p>
      ) : (
        <>
          <p>
            We look at the update of sweep {taken.sweep}, because by then every factor has its own stand-in and step 1
            has something to take out. We take the measurement whose tilted curve is furthest from a bell, so the
            difference can be seen. That is measurement <span className="num">{taken.factor}</span>, with value{' '}
            <span className="num">{fixed(taken.point)}</span>.
          </p>

          <Figure
            wide
            caption={`One update in sweep ${taken.sweep}: the cavity (blue), the real factor scaled to fit (dotted), the tilted curve (orange) and the bell with the same mean and variance (dashed). The gap between orange and dashed is what this step gives up.`}
          >
            <UpdateFigure taken={taken} />
          </Figure>

          <p>
            Before the update the approximation had mean{' '}
            <span className="num">{fixed(beforeBell?.mean ?? Number.NaN)}</span> and sd{' '}
            <span className="num">{fixed(beforeBell?.sd ?? Number.NaN)}</span>. Step 1 removes this factor&rsquo;s own
            stand-in, which had precision <span className="num">{fixed(taken.own.precision)}</span>. What is left, the
            cavity, has mean <span className="num">{fixed(cavityMean)}</span> and sd{' '}
            <span className="num">{fixed(cavitySd)}</span>.
          </p>

          <p>
            Step 2 multiplies the cavity by the real factor. The real factor is the dotted curve: a low floor from the
            clutter explanation and a bump around the measurement from the reading explanation. The product, the orange
            curve, has mean <span className="num">{fixed(taken.tilted.mean)}</span> and sd{' '}
            <span className="num">{fixed(taken.tilted.sd)}</span>. Its mean is{' '}
            <span className="num">{fixed(Math.abs(taken.tilted.mean - cavityMean))}</span>{' '}
            {taken.tilted.mean > cavityMean ? 'above' : 'below'} the cavity&rsquo;s, and its sd is{' '}
            {taken.tilted.sd < cavitySd ? 'smaller' : 'larger'}. It is not a bell, because the real factor is not one.
          </p>

          <p>
            Step 3 is the dashed bell. It has the mean and sd of the orange curve,{' '}
            <span className="num">{fixed(taken.tilted.mean)}</span> and{' '}
            <span className="num">{fixed(taken.tilted.sd)}</span>. In natural parameters that is{' '}
            <Tex>{tex`h = ${fixed(taken.matched.shift)}`}</Tex> and{' '}
            <Tex>{tex`J = ${fixed(taken.matched.precision)}`}</Tex>. The divergence from the orange curve to this bell
            is <span className="num">{fixed(taken.divergence, 3)}</span> nats. It is zero only if the tilted curve is a
            bell.
          </p>

          <p>
            Step 4 subtracts the cavity from the matched bell. The new stand-in has{' '}
            <Tex>{tex`h = ${fixed(taken.standIn.shift)}`}</Tex> and{' '}
            <Tex>{tex`J = ${fixed(taken.standIn.precision)}`}</Tex>
            {taken.standIn.precision > 0
              ? '.'
              : ', a negative precision. The matched bell is wider than the cavity, so the new stand-in has to widen it. It is not a bell, only a correction, and that is allowed because its sum with the cavity is a bell.'}{' '}
            Step 5 puts it in place of the old one. The approximation after the update is the dashed bell.
          </p>

          <Figure
            caption={
              <>
                The same update as numbers, where <Tex>J</Tex> is the precision and the tilted curve has only a mean and
                sd because it is not a bell.
              </>
            }
          >
            <Table
              head={[
                '',
                'mean',
                'sd',
                <>
                  precision <Tex>J</Tex>
                </>,
                <Tex key="h">h</Tex>,
              ]}
              rows={[
                [
                  'approximation before',
                  fixed(beforeBell?.mean ?? Number.NaN),
                  fixed(beforeBell?.sd ?? Number.NaN),
                  fixed(taken.before.precision),
                  fixed(taken.before.shift),
                ],
                [
                  'old stand-in',
                  taken.own.precision > 0 ? fixed(taken.own.shift / taken.own.precision) : '—',
                  taken.own.precision > 0 ? fixed(1 / Math.sqrt(taken.own.precision)) : '—',
                  fixed(taken.own.precision),
                  fixed(taken.own.shift),
                ],
                [
                  '1. cavity',
                  fixed(cavityMean),
                  fixed(cavitySd),
                  fixed(taken.cavity.precision),
                  fixed(taken.cavity.shift),
                ],
                ['2. tilted (not a bell)', fixed(taken.tilted.mean), fixed(taken.tilted.sd), '—', '—'],
                [
                  '3. matched bell',
                  fixed(taken.tilted.mean),
                  fixed(taken.tilted.sd),
                  fixed(taken.matched.precision),
                  fixed(taken.matched.shift),
                ],
                [
                  '4. new stand-in',
                  taken.standIn.precision > 0 ? fixed(taken.standIn.shift / taken.standIn.precision) : '—',
                  taken.standIn.precision > 0 ? fixed(1 / Math.sqrt(taken.standIn.precision)) : '—',
                  fixed(taken.standIn.precision),
                  fixed(taken.standIn.shift),
                ],
              ]}
            />
          </Figure>

          <p>
            The page worked this out itself, with tensor operations on a grid of 401 values of the location (the code is
            at the end). The engine does the same update with its own numerical integration and sends a message. The two
            stand-ins agree: the largest difference between their numbers is within{' '}
            <span className="num">{sci(Math.max(taken.engineGap, 1e-12))}</span>.
          </p>
        </>
      )}

      <h2>5. The sweep</h2>

      <p>
        EP does this update for every factor in turn: the prior first, then measurements 1 to {ROWS}. One pass through
        all {ROWS + 1} is a sweep. After a sweep every stand-in has changed, so the cavities the early factors used are
        out of date. EP sweeps again, and again.
      </p>

      <p>
        In omai this is the engine behind <code>numericEpMethod</code>. The setting{' '}
        <code>granularity: &apos;factor&apos;</code> makes each step of the run one update, so a sweep is {ROWS + 1}{' '}
        steps.
      </p>

      <Figure
        wide
        caption="The belief about the location after each sweep: the mean (line) and one sd either side (band), with the exact mean dashed."
      >
        <SweepsFigure result={result} />
      </Figure>

      <p>
        After sweep 1 the belief had mean{' '}
        <span className="num">{sweeps.first === null ? '—' : fixed(sweeps.first.mean)}</span> and sd{' '}
        <span className="num">{sweeps.first === null ? '—' : fixed(sweeps.first.sd)}</span>. After sweep{' '}
        <span className="num">{result.sweeps}</span>, the last one, it had mean{' '}
        <span className="num">{sweeps.last === null ? '—' : fixed(sweeps.last.mean)}</span> and sd{' '}
        <span className="num">{sweeps.last === null ? '—' : fixed(sweeps.last.sd)}</span>. The exact mean is{' '}
        <span className="num">{fixed(result.exact.mean)}</span>.
      </p>

      <p>
        EP needs a rule for when to stop. After each sweep the engine measures the largest change of any belief, in
        units of its sd. It stops when that is below a tolerance, here {sci(TOLERANCE)}, or after the most sweeps it may
        make, here {MAX_SWEEPS}.
      </p>

      <Figure caption="The largest change of the belief in each sweep, on a log axis, with the tolerance dashed.">
        <ChangeFigure result={result} />
      </Figure>

      {result.settled ? (
        <p>
          This run stopped after <span className="num">{result.sweeps}</span> sweeps, when the largest change was{' '}
          <span className="num">{sci(sweeps.lastChange)}</span>, below the tolerance. Another sweep would move the
          belief by less than that. The stand-ins have reached a fixed point of the updates: each one is the right
          answer given the others.
        </p>
      ) : (
        <p>
          This run did not settle. It made <span className="num">{result.sweeps}</span> sweeps, the most it was allowed,
          and the largest change in the last one was <span className="num">{sci(sweeps.lastChange)}</span>, above the
          tolerance of {sci(TOLERANCE)}. EP is not guaranteed to converge. Part 7 says what the library can do about it.
          Press <em>Run again</em> for other measurements. Most draws settle.
        </p>
      )}

      <h2>6. The same run from start to finish</h2>

      <p>
        Here is the run from the beginning, with the same measurements. At the start every stand-in is flat, so the
        approximation says nothing. The engine updates the prior first. Nothing has been heard yet, so the cavity is
        flat and the tilted distribution is the prior itself, which is a bell. The matched bell and the new stand-in are
        the prior too.
      </p>

      <p>
        The first row of the table shows it: a bell with precision{' '}
        <span className="num">{fixed(firstUpdates[0]?.standIn.precision ?? Number.NaN, 2)}</span>, which is sd{' '}
        {PRIOR_SD}, and mean{' '}
        <span className="num">
          {firstUpdates[0] === undefined
            ? ''
            : fixed(firstUpdates[0].standIn.shift / firstUpdates[0].standIn.precision)}
        </span>
        .
      </p>

      <Figure caption="The first sweep, update by update, with the new stand-in shown as mean (precision). A precision near zero makes the mean meaningless.">
        <Table
          head={['update', 'factor', 'cavity', 'new stand-in', 'approximation after']}
          rows={firstUpdates.map((s, i) => [
            i + 1,
            s.factor === 0 ? 'prior' : yIs(s.factor, result.points[s.factor - 1] as number),
            s.cavity === null ? 'flat' : bellText(s.cavity),
            messageText(s.standIn),
            bellText(s.belief),
          ])}
        />
      </Figure>

      {second === undefined ? null : (
        <p>
          The second update is the first measurement, <span className="num">{fixed(result.points[0] as number)}</span>.
          All the other stand-ins are still flat, so its cavity is the approximation so far, the prior: mean{' '}
          <span className="num">{second.cavity === null ? '—' : fixed(second.cavity.mean)}</span>, sd{' '}
          <span className="num">{second.cavity === null ? '—' : fixed(second.cavity.sd)}</span>. Steps 2 and 3 give a
          matched bell, and step 4 turns it into a stand-in with mean{' '}
          <span className="num">
            {second.standIn.precision > 0 ? fixed(second.standIn.shift / second.standIn.precision) : '—'}
          </span>{' '}
          and precision <span className="num">{fixed(second.standIn.precision)}</span>. The approximation after it has
          mean <span className="num">{second.belief === null ? '—' : fixed(second.belief.mean)}</span> and sd{' '}
          <span className="num">{second.belief === null ? '—' : fixed(second.belief.sd)}</span>.
          {lastUpdate === undefined ? null : (
            <>
              {' '}
              Each later update starts from what the one before left. After the last update of the sweep, the one for
              measurement {lastUpdate.factor}, the approximation has mean{' '}
              <span className="num">{lastUpdate.belief === null ? '—' : fixed(lastUpdate.belief.mean)}</span> and sd{' '}
              <span className="num">{lastUpdate.belief === null ? '—' : fixed(lastUpdate.belief.sd)}</span>. That is the
              end of sweep 1.
            </>
          )}
        </p>
      )}

      <p>
        The second sweep differs, because each factor now has its own stand-in to take out in step 1. The next table
        follows every stand-in over the sweeps. Each cell is the stand-in&rsquo;s mean and, in brackets, its precision.
        The last row is the approximation, the product of the column.
      </p>

      <Figure caption="The stand-in of every factor after some sweeps. A negative precision means it is not a bell, and a precision near zero means it says almost nothing.">
        <Table
          head={['factor', ...sweepTable.map((s) => `after sweep ${s.sweep}`)]}
          rows={[
            ...Array.from({ length: result.rows + 1 }, (_, j) => [
              j === 0 ? 'prior' : yIs(j, result.points[j - 1] as number),
              ...sweepTable.map((s) => messageText(s.messages[j] ?? { shift: 0, precision: 0 })),
            ]),
            ['approximation', ...sweepTable.map((s) => beliefAfter(s.sweep))],
          ]}
        />
      </Figure>

      <p>
        {result.settled
          ? 'The stand-ins change a lot in the first sweeps, and by the last sweep they no longer change.'
          : `In this run the stand-ins had still not stopped changing by sweep ${result.sweeps}.`}
        {sweeps.notBells === 0 ? (
          <> At the end every measurement&rsquo;s stand-in is a proper bell, with positive precision.</>
        ) : (
          <>
            {' '}
            At the end <span className="num">{sweeps.notBells}</span> of the {result.rows} measurement stand-ins{' '}
            {sweeps.notBells === 1 ? 'has' : 'have'} a precision that is not positive.{' '}
            {sweeps.notBells === 1 ? 'It is not a bell' : 'They are not bells'}, and that is fine because the sum with
            the others is.
          </>
        )}{' '}
        {weak.length === 0 ? null : (
          <>
            {' '}
            {weak.length === 1 ? 'The stand-in of' : 'The stand-ins of'}{' '}
            {listNodes(weak.map((j) => <Tex key={j}>{`y_{${j}}`}</Tex>))} {weak.length === 1 ? 'has' : 'have'} a
            precision near zero or below, so {weak.length === 1 ? 'it says' : 'they say'} almost nothing about the
            location: the other measurements agree with each other, and EP has given{' '}
            {weak.length === 1 ? 'this one' : 'these'} little weight. The generator {weakAreClutter ? 'did' : 'did not'}{' '}
            make {weak.length === 1 ? 'it' : 'them'} {weakAreClutter ? '' : 'all '}
            clutter.
          </>
        )}
      </p>

      <p>
        What EP reports is the last row: a bell with mean <span className="num">{fixed(result.ep.mean)}</span> and sd{' '}
        <span className="num">{fixed(result.ep.sd)}</span>.
      </p>

      <p>
        The same run can be watched on the factor graph from part 2. In the replay, each step is one update. The factor
        that was updated sends its new stand-in along its line to the unknown, and the unknown&rsquo;s belief is the
        product of all the stand-ins it holds. Step through the run with the button or the scrubber. The text under the
        graph says what was sent. A message written <Tex>\Gauss(\mu = m, \sigma^2 = v)</Tex> is a bell with mean{' '}
        <Tex>m</Tex> and variance <Tex>v</Tex>.
      </p>

      <MessageReplay
        record={result.record}
        label="The updates of the run, one factor at a time"
        describe={describe}
        caption="The factor graph of the run: the circle is the unknown, the squares are the factors, and a message on a line is the stand-in that factor just sent. The replay keeps the first and last step and the steps where the belief moved most."
      />

      <p>
        Next, the exact answer. The model has one unknown, so the library can compute the posterior on a fine grid of
        values of the location: the product of the prior and the {result.rows} brackets, divided by its total. Here it
        is, with EP&rsquo;s bell on top, the measurements as dots, and the location the data were made around.
      </p>

      <Figure
        wide
        caption="The exact posterior (orange), EP's bell (dashed), the true location (dotted) and the measurements: blue dots are readings and green dots are clutter."
      >
        <AnswerFigure result={result} />
      </Figure>

      <p>
        EP&rsquo;s bell has mean <span className="num">{fixed(result.ep.mean)}</span> and sd{' '}
        <span className="num">{fixed(result.ep.sd)}</span>. The exact posterior has mean{' '}
        <span className="num">{fixed(result.exact.mean)}</span> and sd{' '}
        <span className="num">{fixed(result.exact.sd)}</span>. The means are {fixed(exact.meanGap, 2)} exact sd apart.
        The spread of EP&rsquo;s bell is {fixed(exact.spreadRatio, 2)} times the exact one
        {exact.spreadRatio < 0.95
          ? ', so EP reports more certainty than the exact posterior has.'
          : exact.spreadRatio > 1.05
            ? ', so EP reports less certainty than the exact posterior has.'
            : '.'}
      </p>

      <p>
        {exact.close
          ? 'The two agree closely. Here one bell is a good description of the posterior.'
          : exact.humps > 1
            ? `The exact posterior has ${exact.humps} humps, because the data can be explained in more than one way, and one bell cannot cover them.`
            : 'The exact posterior is not a bell, so no single bell matches it everywhere. This is where EP is an approximation and not the answer.'}{' '}
        {result.settled ? '' : 'This run also did not settle, so the bell shown is where it stopped.'}
      </p>

      {result.logEvidence === undefined ? null : (
        <p>
          EP also estimates the evidence, the total that the posterior is divided by: how probable the model found the
          measurements. It is built from the sizes of the tilted curves met along the way, with no extra computation.
          The run&rsquo;s estimate of the log evidence is <span className="num">{fixed(result.logEvidence, 3)}</span>.
          The grid gives <span className="num">{fixed(result.exact.logEvidence, 3)}</span>, a difference of{' '}
          <span className="num">{fixed(Math.abs(result.logEvidence - result.exact.logEvidence), 3)}</span>.
        </p>
      )}

      <h2>7. What EP gives, and where it goes wrong</h2>

      <p>
        EP gives a bell for each unknown, which is a mean and a spread, and an estimate of the model evidence. It needs
        a number of updates that grows with the number of factors. It also has known weaknesses, and the library has a
        setting for each.
      </p>

      <ul>
        <li>
          <strong>Spreads that are too narrow.</strong> When unknowns are linked in loops, each factor sees a cavity
          that treats them as independent, and the spreads come out too small. The library&rsquo;s compiler packs the
          unknowns that one observation reads into one joint Gaussian vector (<code>packScalars</code>), which keeps
          their correlation. In its test cases, the same models as separate scalars gave spreads two to three times too
          small. This run has one unknown and so no loops. The spread of EP&rsquo;s bell was{' '}
          <span className="num">{fixed(exact.spreadRatio, 2)}</span> times the exact one
          {exact.spreadRatio < 0.95
            ? ', so the spread can come out too small for other reasons too: here, from matching one bell to a curve that is not one.'
            : '.'}
        </li>
        <li>
          <strong>No convergence, and improper messages.</strong> The sweeps can swing instead of settling, and an
          update can give an approximation with a negative variance, which is not a bell. The engine&rsquo;s{' '}
          <code>damping</code> keeps a share of the old message in each update. Its <code>improper</code> policy says
          what to do with an update that has no bell (<code>skip</code>, <code>damp</code>, <code>reset</code> or{' '}
          <code>error</code>). <code>restarts</code> runs again from the start with stronger damping when a run stalls.{' '}
          <code>warmup</code> damps the first three sweeps, which helps models with hidden labels.
        </li>
        <li>
          <strong>One bell for a posterior with two humps.</strong> EP commits to one bell. If the exact posterior has
          two humps, EP covers one of them, or both badly, and which one can depend on the start. In this run the exact
          posterior had <span className="num">{exact.humps}</span> {exact.humps === 1 ? 'hump' : 'humps'}. The
          library&rsquo;s <code>epMultiStartMethod</code> runs EP from several starting points and weights the answers
          by their evidence.
        </li>
        <li>
          <strong>Fractional EP.</strong> The <code>power</code> setting raises each factor to a power between 0 and 1
          in steps 1 and 2. A power of 1 is the EP described here. Smaller powers make each update less eager to follow
          a single factor.
        </li>
      </ul>

      <p>
        The design of EP in omai is written up in <code>docs/probprog/ep.md</code> in the repository.
      </p>

      <h2>8. Where to go next</h2>

      <p>
        The next stories run the same engine on larger models.{' '}
        <Link to={storyHref('two-humps')} className={linkClass}>
          Two kinds of wait at Old Faithful
        </Link>{' '}
        fits a mixture of two bell curves to the waits between eruptions, with a hidden label for each eruption.{' '}
        <Link to={storyHref('when-did-the-rate-change')} className={linkClass}>
          When did the rate change?
        </Link>{' '}
        passes messages along a chain of years to find when a rate changed in a century of mining accidents. And{' '}
        <Link to={storyHref('is-this-coin-fair')} className={linkClass}>
          Is this coin fair?
        </Link>{' '}
        shows beliefs updating one flip at a time, in a model where the exact answer is a formula.
      </p>
    </>
  )
}
