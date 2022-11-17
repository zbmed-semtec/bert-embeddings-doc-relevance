


### [Cosine Similarity Loss](https://www.sbert.net/docs/package_reference/losses.html#cosinesimilarityloss)(CSL)

In cosine similarity loss, the cosine similarity between the query and the Negative document is maximized, while the cosine similarity between the query and the positive document is minimized. The cosine similarity between the embedding pairs is compared with the ground truth relavance score (In our case we get it from the RELISH relevance file). Below table shows the relevance scores from the RELISH relevance document mapped to the cosine similarity scores as per the loss function described in sentence-transformers paper.

| Relevance | RELISH Relevance Score | GT Cosine Similarity Score |
| --- | --- | --- |
| High | 2 | 0.9 |
| partial | 1 | 0.6 |
| low | 0 | 0.3 |

The loss function is defined as:

 $$ v = M(s1)

    u = M(s2)

     S =  (u * v)/(||u|| * ||v||)

     CSL = ||GT - S||_2 $$

Here `v` and `u` = embeddings
    `M` = model
    `S` = cosine similarity
    `GT` = ground truth relevance score
    `s1` and `s2` = query and document pair
    CSL = cosine similarity loss (Means squared error between the ground truth and the cosine similarity)


## [Multiple Negative Ranking Loss](https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss) (MNRL)




