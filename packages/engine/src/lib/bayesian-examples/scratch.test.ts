import { it } from 'vitest'
import { epAtoms } from './ep-atom-list.ts'
import { verifyAtom } from './ep-atoms.ts'
import { atomCalibration } from './ep-extended.ts'

it('scratch', () => {
  const names = (process.env['ATOMS'] ?? '').split(',')
  const cal = process.env['CAL'] === '1'
  const out: unknown[] = []
  for (const a of epAtoms.filter((x) => names.includes(x.name))) {
    const t0 = Date.now()
    if (cal) {
      const c = atomCalibration(a, { seeds: Number(process.env['SEEDS'] ?? 0) || (a.recovery?.seeds ?? 20) })
      out.push({
        name: a.name,
        ms: Date.now() - t0,
        ran: c.ran,
        seeds: c.seeds,
        pass: c.pass,
        rows: c.rows.map((r) => [r.key, r.coverage.toFixed(2), r.zrms.toFixed(2), r.meanZ.toFixed(2), r.pass]),
      })
      continue
    }
    const v = verifyAtom(a)
    out.push({
      name: a.name,
      ms: Date.now() - t0,
      status: v.run.status,
      reason: v.run.status === 'ran' ? '' : v.run.reason,
      pass: v.check?.pass,
      rows: v.check?.rows.map((r) => [r.name, +r.meanZ.toFixed(3), +r.sdRatio.toFixed(3), +r.ks.toFixed(3), r.pass]),
    })
  }
  throw new Error('RES ' + JSON.stringify(out))
})
