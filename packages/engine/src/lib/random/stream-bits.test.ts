// Bit-for-bit pins of the sample streams, recorded from the code before `uniformStream` computed its Philox blocks
// lazily: for a given key the draws must never change. Each row is a checksum (FNV-1a over the 32-bit words of the
// float64 values) of the whole array, and its first value.
import { describe, expect, it } from 'vitest'
import { toFlat } from '../tensor/index.ts'
import { beta, binomial, dirichlet, gamma, gammaVariates, key, poisson, studentT, uniformStream } from './index.ts'

function checksum(values: ArrayLike<number>): string {
  const words = new Uint32Array(Float64Array.from(values).buffer)
  let h = 0x811c9dc5
  for (const w of words) h = Math.imul(h ^ w, 0x01000193) >>> 0
  return h.toString(16).padStart(8, '0')
}

const ALPHAS = Float64Array.from({ length: 64 }, (_, i) => [0.3, 1, 2.5, 30, 0.9, 7][i % 6] ?? 1)

/** The sample behind a pin: `name(seed)` or, for streams, `stream(seed,block,from)`. */
function sampleOf(name: string): ArrayLike<number> {
  const [fn = '', args = ''] = name.slice(0, -1).split('(')
  const [seed = 0, block = 256, from = 0] = args.split(',').map(Number)
  const k = key(seed)
  switch (fn) {
    case 'stream': {
      const next = uniformStream(k, block, from)
      return Float64Array.from({ length: 600 }, () => next())
    }
    case 'gammaVariates':
      return gammaVariates(k, ALPHAS)
    case 'gamma':
      return toFlat(gamma(k, [200], 0.7))
    case 'beta':
      return toFlat(beta(k, [100], 0.5, 2))
    case 'studentT':
      return toFlat(studentT(k, [100], 3))
    case 'dirichlet':
      return toFlat(dirichlet(k, [0.5, 1, 3], 40))
    case 'poisson':
      return toFlat(poisson(k, 4, [200]))
    case 'poissonBig':
      return toFlat(poisson(k, 80, [100]))
    case 'binomial':
      return toFlat(binomial(k, 20, 0.3, [200]))
    case 'binomialBig':
      return toFlat(binomial(k, 1000, 0.4, [100]))
    default:
      throw new Error(`no sample named ${name}`)
  }
}

const PINS: readonly (readonly [string, string, number, number])[] = [
  ['stream(0,1,0)', '8ca4830b', 600, 0.5822938693969723],
  ['stream(0,1,1)', '11a08400', 600, 0.3337522556155773],
  ['stream(0,1,7)', 'bf9c1f5d', 600, 0.05518742098457485],
  ['stream(0,1,300)', 'f6af76c6', 600, 0.5745371449427865],
  ['stream(0,2,0)', '0aa86467', 600, 0.5822938693969723],
  ['stream(0,2,1)', '8d09fa61', 600, 0.975544165934352],
  ['stream(0,2,7)', '40b9413e', 600, 0.6436920867150572],
  ['stream(0,2,300)', 'b4c39545', 600, 0.8114111238349467],
  ['stream(0,3,0)', 'd4d71b94', 600, 0.5822938693969723],
  ['stream(0,3,1)', 'b8a12807', 600, 0.975544165934352],
  ['stream(0,3,7)', '078460e7', 600, 0.9213643863556612],
  ['stream(0,3,300)', 'df856d40', 600, 0.279836505876624],
  ['stream(0,5,0)', '44c17e87', 600, 0.5822938693969723],
  ['stream(0,5,1)', '633320fb', 600, 0.975544165934352],
  ['stream(0,5,7)', '541b6d5d', 600, 0.4752600970367403],
  ['stream(0,5,300)', 'f2220e8b', 600, 0.80443610685021],
  ['stream(0,256,0)', '13c83c94', 600, 0.5822938693969723],
  ['stream(0,256,1)', '20fdf37b', 600, 0.975544165934352],
  ['stream(0,256,7)', '8c315466', 600, 0.045937981311307],
  ['stream(0,256,300)', 'ebe0e6b2', 600, 0.12518681498543405],
  ['gammaVariates(0)', 'dcf3e017', 64, 1.388650128619132],
  ['gamma(0)', '87a2c554', 200, 1.3118007753706977],
  ['beta(0)', '96b405d1', 100, 0.3817010240152741],
  ['studentT(0)', 'c7fa0522', 100, -0.061217365510408385],
  ['dirichlet(0)', '15acc2ce', 120, 0.16948009873221231],
  ['poisson(0)', 'b71a6705', 200, 3],
  ['poissonBig(0)', 'febff865', 100, 67],
  ['binomial(0)', 'ab486705', 200, 4],
  ['binomialBig(0)', 'cf0ed865', 100, 396],
  ['stream(7,1,0)', '9336570c', 600, 0.7238820552710848],
  ['stream(7,1,1)', '7bb4e6db', 600, 0.6805837724811784],
  ['stream(7,1,7)', '94a1861f', 600, 0.06369988874719978],
  ['stream(7,1,300)', 'aedb7fd8', 600, 0.40467441080314015],
  ['stream(7,2,0)', 'e0b0fb72', 600, 0.7238820552710848],
  ['stream(7,2,1)', '7673c9dc', 600, 0.5937725828809671],
  ['stream(7,2,7)', '0a5df2f2', 600, 0.7769041544930443],
  ['stream(7,2,300)', 'dc907a8a', 600, 0.9778850222926357],
  ['stream(7,3,0)', '31ff26ce', 600, 0.7238820552710848],
  ['stream(7,3,1)', 'd493694a', 600, 0.5937725828809671],
  ['stream(7,3,7)', '726865d4', 600, 0.08004732994281338],
  ['stream(7,3,300)', 'adf60d48', 600, 0.14251664675811748],
  ['stream(7,5,0)', '58aad6df', 600, 0.7238820552710848],
  ['stream(7,5,1)', '7e31a85a', 600, 0.5937725828809671],
  ['stream(7,5,7)', '777e4173', 600, 0.8518712905955664],
  ['stream(7,5,300)', '07b608a9', 600, 0.40822612679323245],
  ['stream(7,256,0)', '91d52ab3', 600, 0.7238820552710848],
  ['stream(7,256,1)', '169dfd69', 600, 0.5937725828809671],
  ['stream(7,256,7)', '20737962', 600, 0.1599261954235185],
  ['stream(7,256,300)', 'ab9309cd', 600, 0.38466819926017803],
  ['gammaVariates(7)', 'a5dd5a82', 64, 0.015861896680236184],
  ['gamma(7)', '543ded08', 200, 0.5413184470718303],
  ['beta(7)', '6f1134aa', 100, 0.03831418951080337],
  ['studentT(7)', '4ee65158', 100, -0.5786506916643104],
  ['dirichlet(7)', 'dfd864f2', 120, 0.02439513536755501],
  ['poisson(7)', 'd5766705', 200, 7],
  ['poissonBig(7)', 'ac13f865', 100, 86],
  ['binomial(7)', '2a5e6705', 200, 4],
  ['binomialBig(7)', 'f6553865', 100, 421],
  ['stream(12345,1,0)', '0fa65189', 600, 0.7580371219075386],
  ['stream(12345,1,1)', 'fb9834c7', 600, 0.6387318957347078],
  ['stream(12345,1,7)', 'a408a7f9', 600, 0.05241042025425702],
  ['stream(12345,1,300)', '994681ae', 600, 0.6070462786477624],
  ['stream(12345,2,0)', '64e5b15b', 600, 0.7580371219075386],
  ['stream(12345,2,1)', '470455ac', 600, 0.2406184054983228],
  ['stream(12345,2,7)', '3b75d0e9', 600, 0.09276549665478595],
  ['stream(12345,2,300)', '86f3f6ca', 600, 0.8498821271394302],
  ['stream(12345,3,0)', '550d4058', 600, 0.7580371219075386],
  ['stream(12345,3,1)', 'dfbba511', 600, 0.2406184054983228],
  ['stream(12345,3,7)', '641f0931', 600, 0.9835732549678612],
  ['stream(12345,3,300)', 'e79d88cb', 600, 0.07247624756642634],
  ['stream(12345,5,0)', '68c8c74e', 600, 0.7580371219075386],
  ['stream(12345,5,1)', '852c0890', 600, 0.2406184054983228],
  ['stream(12345,5,7)', '257bae1c', 600, 0.8904323934091785],
  ['stream(12345,5,300)', '67d1629f', 600, 0.3389899170809202],
  ['stream(12345,256,0)', '68d307c2', 600, 0.7580371219075386],
  ['stream(12345,256,1)', 'e35f9466', 600, 0.2406184054983228],
  ['stream(12345,256,7)', '96dc8087', 600, 0.9697618977991801],
  ['stream(12345,256,300)', 'f26dd6d7', 600, 0.46338150483771756],
  ['gammaVariates(12345)', '74267a26', 64, 0.6131013551820811],
  ['gamma(12345)', '664acff7', 200, 0.945287147253217],
  ['beta(12345)', '03be84d8', 100, 0.5225716409470544],
  ['studentT(12345)', 'cecdaef8', 100, 4.110508244416309],
  ['dirichlet(12345)', 'f9c68f95', 120, 0.2174947839092428],
  ['poisson(12345)', 'c8506705', 200, 4],
  ['poissonBig(12345)', '70513865', 100, 87],
  ['binomial(12345)', 'd2a06705', 200, 7],
  ['binomialBig(12345)', '84651865', 100, 386],
]

describe('sample streams are bit-for-bit stable', () => {
  it.each(PINS)('%s', (name, hash, length, first) => {
    const values = sampleOf(name)
    expect(values.length).toBe(length)
    expect(values[0]).toBe(first)
    expect(checksum(values)).toBe(hash)
  })
})
