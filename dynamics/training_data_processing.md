This readme introduces how to process tracking data and raw data to obtain the dataset used for training.

1. Run `post_process_tracking_data.py` to extract relavent fields from tracking output and raw data to form a dataset. The output will be a folder of `.h5` files. See  `post_process_commands.txt` for some command examples.
2. Annotate the dataset. Specifically, each dictionary needs a field `object_cls` which represents object category. This is useful sometimes, e.g., we want to analyze the relation between the learned physics params and the box class. Otheriwse, can just set it to -1. The script `fix_training_data_format.py` does this. The other thing that it does is that, it pad zeros to the object point array to make all sequences have the same number of objects. This is a quick fix that could facilitate model training.
3. Lastly, we need to compute the statistics of the dataset. Run `compute_data_stats.py` for that.

