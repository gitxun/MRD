Mutual Refinement Distillation (MRD) for Multimodal Emotion Recognition
Overview
With the rapid advancement of speech emotion recognition, the transition from unimodal to multimodal approaches has become inevitable. However, multimodal methods introduce new challenges, particularly classification ambiguity in complex samples when compared to unimodal approaches.

To address these challenges, we propose a Mutual Refinement Distillation (MRD) method, which integrates three key components to enhance multimodal emotion recognition:

Modal Interaction Calibration

Enhances classification accuracy for complex samples by calibrating the interactions between different modalities.

Interactive Learning Constraints

Mitigates overfitting by enforcing constraints during the interactive learning process.

Reverse Curriculum Learning

Further improves model robustness by adjusting the learning sequence, starting from more difficult samples.

Methodology
The MRD framework is designed to jointly refine multimodal representations and facilitate effective knowledge distillation between modalities, specifically targeting the classification of complex and ambiguous samples in multimodal emotion recognition tasks.

Pipeline
<img width="1282" height="702" alt="pipline" src="https://github.com/user-attachments/assets/1e64308f-281f-463f-9a29-e4abcbb8c634" />

Experiments
We evaluate the MRD method on two widely-used benchmark datasets:

MELD
IEMOCAP
Our experimental results demonstrate that MRD outperforms state-of-the-art methods in emotion recognition, achieving a notable 6.07% improvement over the baseline on the IEMOCAP dataset.
