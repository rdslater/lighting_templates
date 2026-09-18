# Templates
Using this for reproducible code within my group.  Models and Data can be configured but the training and predict scripts will help standardize as different people run others models.  

Note that predict is a placeholder.

TODO: Put in predict that aggregates predictions and saves as a CSV with link to original image (we deal in images), original target (or link) and prediction (or link).

Right now monitors val-loss and has early stopping set at 5.  Obviously configurable via the YAML file.

At some point I think I want to copy the YAML file used and put it in the checkpoint directory.
