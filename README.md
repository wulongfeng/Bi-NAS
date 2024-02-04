# ExplainableRec with Bi-NAS
This is the implmentation for paper: Towards Effective Explanation for Recommender Systems via Bi-Level Neural Architecture Search

# Introduction
Recommender systems have emerged as indispensable tools for helping users navigate and manage the overwhelming influx of information. While these systems strive to offer personalized and relevant suggestions, users may remain unaware of the rationale behind these recommendations, and an explanation of why certain items are recommended to users plays a pivotal role in the process of decision-making. An effective explanation can improve users' experience, bolster their trust, and increase their engagement and loyalty to recommender systems. However, traditional explainable models often exhibit limited interpretability performance, while most modern methods demand significant time and expert knowledge for the intricate construction of explainable pathways, attention mechanisms, and related components.

To enable effective explanations for the recommender system, we introduce a Bi-level Neural Architecture Search (Bi-NAS) framework, which can jointly discover the optimal cross-attention construction and feature interaction functions. Specifically, we define a search space that encompasses both \emph{intra-layer design} and \emph{inter-layer design}. Intra-layer pertains to the construction of cross-attention mechanisms, while inter-layer focuses on the interaction functions that operate on the users/items and their corresponding features. This dual design empowers our recommender system to dynamically adapt and optimize its performance across diverse recommendation scenarios. Moreover, by harnessing cross-attention, we gain access to users' preferences for features and the quality of items with respect to those features. Effective and insightful explanations can be attained when the weight assigned to users' feature preferences aligns with the weight representing the quality of items on those specific features. We perform extensive experimental evaluations using our proposed framework on four real-world datasets and demonstrate its superior performance in terms of recommendation and explanation both quantitatively and qualitatively. 

# Example to run proposed method
For the pre-trained word representation model, we utilize glove.6B.300d.txt, please download it [here](https://nlp.stanford.edu/projects/glove/) and put it in the data folder.

```bash
./example.sh
```

