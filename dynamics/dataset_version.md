# Dataset version control
This file intends to record all our dataset versions for easy tracking. Record for the earlier versions might have been missing.

| Dataset version | Subversion | Notes                                                                                                                                                                                 |
|-----------------|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| v13 | - | Data of six boxes (box1-box6), 5 highly controlled trajectories for each. Seeking to understand the benefit of touch with controlled settings  |
| v14 | - | Data of six boxes (box1-box6), 15min random interaction data for each. Seeking to benchmark different models  |
| v14 | v2 | Post-processed by keeping all 50 points of the box and the rod, as well as the soft bubble  |
| v18 | v1 | Data of two boxes (box3 and box7), 45min data collection for each. Seeking to understand the benefit of touch |
| v18             | v2         | Data smoothed out during parsing to make it cleaner. Use pre-defined masks to sample 20 points for both soft bubbles and objects.                                                                        |
| v18             | v2         | Use fps to sample softbubble points (50) and keep all object points (50) in post-processing.      During training, 20 points are sampled independently with fps for every trajectory. This is to make the data more randomized. |
|                 |            |                                                                                                                                                                                       |
