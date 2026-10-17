# Supabase Grad-CAM failure analysis

Source: `official-supabase-1`, `full_improved-seed0`, run ID 4, dataset ID 3. Regenerated test accuracy: 94.00% (376 correct and 24 incorrect out of 400), matching the stored result.

Evidence: [incorrect examples](../experiments/results/official-supabase-1-gradcam/gradcam_incorrect.png), [correct examples](../experiments/results/official-supabase-1-gradcam/gradcam_correct.png), [example predictions](../experiments/results/official-supabase-1-gradcam/examples.csv), and [source metadata](../experiments/results/official-supabase-1-gradcam/source.json).

## Example 924: horizontal → diagonal, 95% confidence — dataset difficulty

The true image contains a horizontal bar along the top edge. Heavy noise breaks its appearance into bright and dark fragments. The predicted-class Grad-CAM highlights parts of that top region, but the model still predicts diagonal. This suggests the model located relevant evidence but could not reliably distinguish its orientation under severe noise. The true-class heatmap also highlights the top bar. This is consistent with dataset difficulty caused by corrupted visual information, although the model’s feature representation may also contribute.

## Example 1783: horizontal → block, 59% confidence — model limitation

The true horizontal bar lies near the top edge, but the predicted-block heatmap responds most strongly to broad patches in the middle and lower background. In contrast, the true-horizontal heatmap emphasizes the top bar. This suggests the model relied on misleading background patterns rather than the actual shape when assigning its highest score. The example supports a model limitation: sensitivity to spurious features in noisy images.

## Interpretation limits

Grad-CAM shows regions associated with a class score; it does not prove the cause of an error. These examples suggest different failure mechanisms—obscured shape information versus reliance on background patterns. They also show that confidence alone is insufficient: example 924 was incorrect despite 95% confidence.
