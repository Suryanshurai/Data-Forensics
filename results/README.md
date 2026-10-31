# Results

`cnn_results.zip` contains the CNN outputs from the Colab notebook (DF (2).ipynb):

- cnn_0p1_test_predictions.csv and cnn_0p4_test_predictions.csv: test-set scores and predictions (100 originals and 100 stego images per payload).
- cnn_0p1_best.pt and cnn_0p4_best.pt: saved CNN weights (lowest validation loss).

Test AUC recomputed from these files: 0.662 at 0.1 bpp, 0.931 at 0.4 bpp.
