# TBKCF
A brief introduction of our methods.
The paper is under review.

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


The notes for the folders:
The folder '.\codes' contains the codes of our method.
The folder '.\data' contains the experimental results of our method.
The folder '.\figs and tables' contains the figs and tables of our paper.

Please cite this article
Liu, C., Liu, P., Zhao, W., & Tang, X. (2017). Robust tracking and re-detection: collaboratively modeling the target and its context. IEEE Transactions on Multimedia.
