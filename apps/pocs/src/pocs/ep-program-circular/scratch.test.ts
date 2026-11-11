import { toFlat } from 'omai-engine/tensor'
import { it } from 'vitest'
import { infer, simulate, type TruthName, type Variant } from './program.ts'

it('probe', () => {
  const N = Number(process.env['NN'] ?? 200)
  for (const [truth, K] of [['one', 1], ['two', 2]] as [TruthName, number][]) {
    for (const variant of ['ignore', 'means', 'data', 'rows'] as Variant[]) {
      const y = simulate(1, truth, N)
      const t0 = performance.now()
      let line = `${truth} K${K} ${variant}: `
      try {
        const { run, posterior } = infer(variant, K, y)
        const f = (s: string) => Array.from(toFlat(posterior.mean(s))).map((v) => v.toFixed(2)).join('/')
        const sd = 1 / Math.sqrt(Number(toFlat(posterior.mean('tau'))[0]))
        line += `${run.status} ${run.t} ${Math.round(performance.now() - t0)}ms mean ${f('mean')} sd ${sd.toFixed(2)} ${K > 1 ? 'w ' + f('weight') : ''}`
      } catch (e) { line += 'ERR ' + String(e).slice(0, 200) }
      console.log(line)
    }
  }
}, 900000)
