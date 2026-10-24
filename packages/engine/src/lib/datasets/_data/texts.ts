// Small public-domain texts embedded by hand: traditional nursery rhymes and two textbook worked-example corpora.

/** Traditional English nursery rhymes (public domain), first verses. */
export const NURSERY_RHYMES: readonly { readonly title: string; readonly text: string }[] = [
  {
    title: 'Mary Had a Little Lamb',
    text: 'Mary had a little lamb, its fleece was white as snow; and everywhere that Mary went, the lamb was sure to go.',
  },
  {
    title: 'Twinkle, Twinkle, Little Star',
    text: 'Twinkle, twinkle, little star, how I wonder what you are! Up above the world so high, like a diamond in the sky.',
  },
  {
    title: 'Humpty Dumpty',
    text: "Humpty Dumpty sat on a wall, Humpty Dumpty had a great fall. All the king's horses and all the king's men couldn't put Humpty together again.",
  },
  {
    title: 'Jack and Jill',
    text: 'Jack and Jill went up the hill to fetch a pail of water. Jack fell down and broke his crown, and Jill came tumbling after.',
  },
  {
    title: 'Baa, Baa, Black Sheep',
    text: 'Baa, baa, black sheep, have you any wool? Yes sir, yes sir, three bags full; one for the master, one for the dame, and one for the little boy who lives down the lane.',
  },
  {
    title: 'Hey Diddle Diddle',
    text: 'Hey diddle diddle, the cat and the fiddle, the cow jumped over the moon; the little dog laughed to see such sport, and the dish ran away with the spoon.',
  },
  {
    title: 'Little Miss Muffet',
    text: 'Little Miss Muffet sat on a tuffet, eating her curds and whey; along came a spider, who sat down beside her and frightened Miss Muffet away.',
  },
  {
    title: 'Hickory Dickory Dock',
    text: 'Hickory dickory dock, the mouse ran up the clock. The clock struck one, the mouse ran down, hickory dickory dock.',
  },
  {
    title: 'Little Bo-Peep',
    text: "Little Bo-Peep has lost her sheep, and doesn't know where to find them; leave them alone, and they'll come home, wagging their tails behind them.",
  },
  {
    title: 'Three Blind Mice',
    text: "Three blind mice, three blind mice, see how they run, see how they run! They all ran after the farmer's wife, who cut off their tails with a carving knife.",
  },
  {
    title: 'Little Jack Horner',
    text: 'Little Jack Horner sat in the corner, eating his Christmas pie; he put in his thumb, and pulled out a plum, and said, what a good boy am I!',
  },
  {
    title: 'Georgie Porgie',
    text: 'Georgie Porgie, pudding and pie, kissed the girls and made them cry; when the boys came out to play, Georgie Porgie ran away.',
  },
  {
    title: 'Mary, Mary, Quite Contrary',
    text: 'Mary, Mary, quite contrary, how does your garden grow? With silver bells and cockle shells, and pretty maids all in a row.',
  },
  {
    title: 'Hush-a-bye Baby',
    text: 'Hush-a-bye, baby, on the tree top, when the wind blows the cradle will rock; when the bough breaks, the cradle will fall, and down will come baby, cradle and all.',
  },
  {
    title: 'Pat-a-cake',
    text: "Pat-a-cake, pat-a-cake, baker's man, bake me a cake as fast as you can; pat it and prick it and mark it with B, and put it in the oven for baby and me.",
  },
  {
    title: 'Row, Row, Row Your Boat',
    text: 'Row, row, row your boat, gently down the stream. Merrily, merrily, merrily, merrily, life is but a dream.',
  },
]

/** The nine technical memo titles of Deerwester et al. (1990), the worked example of latent semantic analysis. */
export const DEERWESTER_TITLES: readonly {
  readonly id: string
  readonly text: string
  readonly label: 'hci' | 'graphs'
}[] = [
  { id: 'c1', text: 'Human machine interface for ABC computer applications', label: 'hci' },
  { id: 'c2', text: 'A survey of user opinion of computer system response time', label: 'hci' },
  { id: 'c3', text: 'The EPS user interface management system', label: 'hci' },
  { id: 'c4', text: 'System and human system engineering testing of EPS', label: 'hci' },
  { id: 'c5', text: 'Relation of user perceived response time to error measurement', label: 'hci' },
  { id: 'm1', text: 'The generation of random, binary, ordered trees', label: 'graphs' },
  { id: 'm2', text: 'The intersection graph of paths in trees', label: 'graphs' },
  { id: 'm3', text: 'Graph minors IV: Widths of trees and well-quasi-ordering', label: 'graphs' },
  { id: 'm4', text: 'Graph minors: A survey', label: 'graphs' },
]

/** The two Shakespeare passages of Manning, Raghavan and Schütze's inverted-index example (Introduction to IR, ch. 1). */
export const CAESAR_PASSAGES: readonly string[] = [
  "I did enact Julius Caesar: I was killed i' the Capitol; Brutus killed me.",
  'So let it be with Caesar. The noble Brutus hath told you Caesar was ambitious:',
]
