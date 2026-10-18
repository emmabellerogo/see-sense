# Engineering report: Komal Khan

## My role
My job was to make sure our comparison was fair and that every number could be checked. I wrote the scripts for exploring settings, analysing results, checking the SQL tables and producing the Grad-CAM figures, and I wrote the first version of our experiment protocol. The later switch from synthetic shapes to Fashion-MNIST was done in an AI-assisted session on Emma's machine,

## Experiment
- **I never chose settings by looking at the test set.** While exploring, I used separate seeds and only looked at validation results.
- **We fixed the plan before running.** Seeds, epochs and data split were written down first, and the run would stop if anything changed.
- **We changed one thing at a time.** Besides the plain baseline and the full improved model, we trained four versions that each remove one technique (batch norm, dropout, augmentation or the learning-rate schedule). All six used the same data and settings for each seed, so we could compare them directly.
- **We checked the numbers twice.** The results tables from Supabase matched an earlier local run exactly.

## What we found
| Model | Test accuracy (average of 3 seeds) |
|---|---|
| No augmentation | 90.30% |
| Baseline | 89.73% |
| No dropout | 88.94% |
| No scheduler | 88.23% |
| Full improved | 88.10% |
| No batch norm | 87.12% |

The surprise is that **the "improved" model did worse than the plain baseline**. Batch norm was the only technique that helped. Augmentation and dropout actually lowered accuracy, and the learning-rate schedule made no real difference.

My best explanation is that the model is too small (about 9,000 parameters). The improved model was not overfitting, since its training and validation accuracy were both around 89%, so the extra regularization only held it back. This is a guess we did not test. With only three seeds, small differences between models should not be over-read.

## Where the model struggles
- **Shirt is the hardest class.** It gets mixed up with T-shirts, coats and pullovers, which all have similar shapes. Trousers, bags, sandals and ankle boots are easy (over 93%).
- **Grad-CAM has limits.** When the model confuses shoe types with high confidence, the images themselves look ambiguous at 28×28. For most Dress and Coat predictions, Grad-CAM shows nothing at all, so it cannot explain those decisions.
- **White backgrounds confuse it.** Photos with a light background were mostly predicted as "Bag", often with high confidence.

## What I would do differently
- Check the assignment's dataset requirement at the start. We built on the starter's synthetic shapes and found out late that Fashion-MNIST was required.
- Use more seeds and try a bigger model, to find out whether the techniques really hurt or the model is just too small.
- Test on real-world images much earlier.

## Christian worldview
Our dataset, Fashion-MNIST, has a background bias. Every training image shows a light clothing item on a dark background, so the model only learned what clothing looks like in that setting. When we turned the background of the test images white, 8 of 9 items were wrongly predicted as "Bag", some with almost 100% confidence. The model only works properly on images with a dark background.

**Bias** Every person is made in the image of God (Genesis 1:27), and God shows no partiality (Acts 10:34). Our model is harmless, but the same mistake in a real system, like face recognition or medical imaging trained mostly on one group of people, would serve some people well and fail others. Micah 6:8 calls us to "do justly", and for an engineer that means asking who our data leaves out.

**Honesty** Proverbs 11:1 says "a false balance is abomination to the LORD: but a just weight is his delight." A model that confidently gives the wrong answer is like a false balance: it looks trustworthy but isn't. That is why we openly report this bias in our model card and warn users in the app.

**Humility** First Corinthians 13:12 says we "see through a glass, darkly." Our model sees even less, since it only knows the images it was given. Remembering that keeps us from trusting a confident output more than it deserves.

This project taught me that a model is only as fair as its data, and that being a Christian engineer means being truthful about where a system fails, not just making it work.

## AI use
My AI sessions are recorded in `ai-usage/Komal-Session1.md` and `Komal-Session2.md`. I also used an AI assistant to help draft and simplify this report, including the worldview section. [Komal to add what you checked yourself]ve done without the assistant.]
