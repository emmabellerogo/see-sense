# Draft question for the instructor — NOT SENT

*Draft for Emma (and Komal) to edit and send, or not. Nothing here has been sent, and no answer or approval exists.*

**Subject:** See-Sense (Product 3): dataset requirement and our status

Hello Professor,

While finishing See-Sense we found that our project was built on the synthetic four-shape dataset that came with the starter code (horizontal bar, vertical bar,
diagonal line, filled block), not on CIFAR-10 or Fashion-MNIST as the assignment states. We have now moved the model, API, database schema and UI to
**Fashion-MNIST** (all ten classes, the official train/test split, a validation set carved from the training images) and have re-run the baseline, improved
and four leave-one-technique-out experiments on it (3 seeds each, 18 runs).

Two quick questions, so we do not guess:

1. Is it acceptable to keep the earlier synthetic-shape experiment in the repository, clearly labelled "historical, not the final evidence", or would you prefer that it be removed from the submission?
2. Our compute is a laptop CPU. A full six-variant comparison takes about 20 minutes of recorded training time per training seed (about 0.97 h for all 18 runs, measured; one run takes 2 to 5 minutes). Is it acceptable to report 3 (the number we used) training seeds per variant (we will state the number and its limits plainly), rather than a larger number?

If the deadline or grading expectations differ from what we assumed, please tell us. Thank you.

Emma Rogoveanu and Komal Khan
