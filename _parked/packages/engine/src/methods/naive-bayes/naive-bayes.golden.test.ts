import { sliceAxis, type Tensor } from '../../lib/tensor/index.ts'
import { learnGolden, num, str } from '../../testing/learn.ts'
import { bernoulliNB, categoricalNB, complementNB, gaussianNB, multinomialNB } from './naive-bayes.ts'

const makers = {
  gaussian: () => gaussianNB(),
  multinomial: (a: number) => multinomialNB({ alpha: a }),
  bernoulli: (a: number) => bernoulliNB({ alpha: a, binarize: 1 }),
  complement: (a: number) => complementNB({ alpha: a }),
  categorical: (a: number) => categoricalNB({ alpha: a }),
} as const

learnGolden('naive Bayes against scikit-learn', 'methods/naive_bayes/naive_bayes', {
  nb: (inputs, p) => {
    const [x, y] = inputs as readonly [Tensor, Tensor]
    const kind = str(p, 'kind') as keyof typeof makers
    const f = makers[kind](num(p, 'alpha', 1)).fit({ x, y })
    const head = sliceAxis(x, 0, 0, 6)
    return { jll: f.predictJointLogProba(head), proba: f.predictProba(head) }
  },
})
