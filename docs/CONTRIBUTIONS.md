# Contribution statements and individual videos

**Placeholders only.** Nothing below describes what either person did or believes; each person
must write their own statement and record their own video. Commit history and `ai-usage/`
records exist, but they are not a substitute for a personal statement.

## Emma Rogoveanu

**Contribution statement:**
> TODO (Emma): describe in your own words what you built, decided and learned. Mention which
> parts of the work were yours and which were done jointly.

**Individual video link:** [Emma’s video walkthrough](https://www.loom.com/share/8a3f95c3eb824a01a19407af70d2c41c)

**Disclosure reflection:** see the draft in [ETHICS_AND_DISCLOSURE.md](ETHICS_AND_DISCLOSURE.md) §6 (for Emma to review).

## Komal Khan

**Contribution statement:**
I was responsible for the experiment and the evidence. I fixed setup before the official run (Fashion-MNIST, 20 epochs, batch size 128, learning rate 0.001, seeds 0–2), ran all six configurations across three seeds (18 runs), saved them to Supabase and retrieved the results with SQL. Also preparing the Grad-CAM evidence and explained two failures. The switch to Fashion-MNIST was done by Emma. I learned that fair experiment matters more than a high score, our "improved" model did worse than the baseline, and that was still worth reporting.

**Individual video link:** [https://www.loom.com/share/d341818a31304cdb9add752bd470809a]

## Dataset correction (2026-10-17)
The move from the synthetic shapes to Fashion-MNIST (branch `correction/fashion-mnist`) was done in an AI-assisted session on Emma's
machine and repository. It is **not attributed to Komal**, and it does not replace either person's own statement. Komal's and Emma's earlier
work (see the records below) is unchanged. Each person still writes, in their own words, what they did, decided and learned.

## Supporting records (to cite, not to replace the statements)
- AI-usage records: [Emma Session 1](../ai-usage/Emma-Session1.md), [Emma Session 3](../ai-usage/Emma-Session3.md), [Komal Session 1](../ai-usage/Komal-Session1.md), [Komal Session 2](../ai-usage/Komal-Session2.md)
- Git history: `git log --format='%an %s'`
