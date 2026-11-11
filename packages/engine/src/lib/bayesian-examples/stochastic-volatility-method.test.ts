import fs from 'node:fs'
import { describe, expect, it } from 'vitest'
import { defaults } from '../space/index.ts'
import { type Config, session } from '../trace/index.ts'
import { volatilityExample } from './stochastic-volatility-method.ts'

const out: string[] = []
const log = (s: string): void => {
  out.push(s)
  fs.writeFileSync(
    '/private/tmp/claude-501/-Users-nialltwomey-workspace-whitebox-js/d4034594-54d3-484e-a26a-331816a1f3f5/scratchpad/sv-run.txt',
    out.join('\n'),
  )
}
const base = defaults(volatilityExample.method.config) as Config
function run(label: string, over: Config, inference: Config): void {
  const config = {
    ...base,
    ...over,
    inference: {
      ...(defaults(
        volatilityExample.offer({ ...base, ...over }).space.options[inference['method'] as string] as never,
      ) as Config),
      ...inference,
    },
  }
  const t0 = Date.now()
  try {
    const s = session(volatilityExample.method, { config, steps: volatilityExample.steps(config), mode: 'eager' })
    const t1 = Date.now()
    const scores = s.probe(s.t, 'sv/scores')
    log(
      label +
        ' run ' +
        (t1 - t0) +
        'ms probes ' +
        (Date.now() - t1) +
        'ms steps ' +
        s.t +
        ' ' +
        String(s.status) +
        ' ' +
        JSON.stringify(scores),
    )
  } catch (e) {
    log(label + ' ERROR ' + String(e).slice(0, 400))
  }
}
describe('scratch', () => {
  it('runs', () => {
    run('T60 nuts dense 1x(80+100)', { length: 60 }, { method: 'nuts', numChains: 1, numWarmup: 80, numDraws: 100, dense: true })
    run('T60 nuts diag 1x(80+100)', { length: 60 }, { method: 'nuts', numChains: 1, numWarmup: 80, numDraws: 100 })
    expect(1).toBe(1)
  }, 600000)
})
