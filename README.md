# TBKCF

Robust Tracking and Re-detection:
Collaboratively Modeling the Target and Its Context

Abstract—Robust object tracking and re-detection require stably predicting the trajectory of the target object and recovering
from tracking failure by quickly re-detecting it when it is lost during long-term tracking. The locations of the target and the
background are calculated relative to the region occupied by the object. The effect of tracking can be enhanced by isolating
the target and the background, modeling and tracking them, respectively, and integrating their tracking results. In this study,
we propose an approach that builds motion models for the target and its context. Tracking results from a target tracker and a
context tracker are integrated through linear fusion to predict the position of the target. A kernelized correlation filter (KCF)
tracker is used to track the target in the predicted position. When the target is lost, it can be quickly recovered by searching
in the given field of view using a target model built and updated through observation models that are constructed prior to the loss
of the target. Our approach is not sensitive to the segmentation of the target and the context. The motion models and observation
models of the target and the context work together in the tracking process, whereas the target model alone is involved in
re-detection. Experiments to test our proposed approach, which simultaneously models the target and its context, showed that it
can effectively enhance the robustness of long-term tracking. 
Index Terms—Collaborative modeling, Fusion of tracking results, Re-detection, Target tracking.


Please cite this paper
@article{DBLP:journals/tmm/LiuLZT18,
  author    = {Chang Liu and
               Peng Liu and
               Wei Zhao and
               Xianglong Tang},
  title     = {Robust Tracking and Redetection: Collaboratively Modeling the Target
               and Its Context},
  journal   = {{IEEE} Trans. Multimedia},
  volume    = {20},
  number    = {4},
  pages     = {889--902},
  year      = {2018},
  url       = {https://doi.org/10.1109/TMM.2017.2760633},
  doi       = {10.1109/TMM.2017.2760633},
  timestamp = {Wed, 11 Apr 2018 09:53:32 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/tmm/LiuLZT18},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
