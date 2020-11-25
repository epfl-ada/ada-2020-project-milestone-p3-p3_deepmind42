# The Reddit hyperlink network, balance versus status theory

## Abstract
*A 150 word description of the project idea, goals, datasets used. What's the motivation behind your project? How do you propose to extend the analysis from the paper? What story would you like to tell, and why?*

The goal of the project is to explore the Reddit hyperlink network
and in particular to assess if the balance and the status theories can be applied for this dataset.
Also, we examine if the network exhibit locally the same behavior as the whole network. We will use the embedding vectors of the subreddits to group them into clusters,
 and check if the balance theory prevails over the status one inside these clusters. Also, there might be clusters for which the dominant theory is different than for some others.
One final important aspect is the temporal dimension of the network.
We propose to analysis the evolution of the network using the balance and status theories, which might reveal structural changes as the network grows (the addition of new edges or the modification of the sign of a previous one).

## Research Questions
*A list of research questions you would like to address during the project.*

- Do the status and balance theories apply to the proposed dataset (the Reddit hyperlink network)? If not, why ?
- Regardless of whether these two theories can be applied to the whole dataset, is it possible to carry out the same analysis on more local parts of the network that share common similarities ? 
- Can we detect structural changes through time using balance and status theories on the whole network ? Do we have a phase transition?

## Proposed dataset
*List the dataset(s) you want to use, and some ideas on how you expect to get, manage, process, and enrich it/them. Show us that you've read the docs and some examples, and that you have a clear idea on what to expect. Discuss data size and format if relevant. It is your responsibility to check that what you propose is feasible given the datasets at hand.*

We propose to use the reddit dataset (which can be found here: https://snap.stanford.edu/data/soc-RedditHyperlinks.html). This dataset contains the hyperlink network representing the connection between subreddits, from January 2014 to April 2017. The nodes are represented by the different subreddits, and each edge corresponds to the hyperlink between subreddits appearing in each post. Therefore an edge might appear several times and we need to figure out how to assign a timestamp to each one of them. Without double-counting the edges that appears multiple times, there are a total of 55863 nodes, and 858490 edges. The network is directed and signed. The latter were obtained through sentimental analysis, and takes value in {-1,+1}. Also, the graph has a temporal dimension that keeps track of the date at which each hyperlink was created.
Another complementary dataset of the reddit hyperlink network can be found at http://snap.stanford.edu/data/web-RedditEmbeddings.html. It gives the embedding vectors for each subreddit. The embedding is of size 300, and there are a total of 51278 vectors (so we might exclude some subreddit in the analysis).

## Methods
- *Whole dataset analysis: balance versus status theory.* We will perform similar analysis as described in the Signed Network paper. In particular, each configuration for the (un)directed triads will be counted and the proportions will be compared to what predict the two theories. We need to implement an algorithm to find these proportions that takes into account the timestamps.
- *Clustering the dataset:* Using the embeddings, we might use the K-means algorithm to get the subnetworks, and the choice for the number of clusters could be done with elbow method, or with the gap statistic. Another possibility for getting the clusters would be to use a Gaussian mixture model.
- *Temporal analysis*: as the networks evolves through time, we can analysis the relative proportions of each type of triads for each network given at some point in time, from 2014 to 2017. This can be however computationally expensive, and we might also need to reduce the number of networks analyzed, either by only selecting networks that resulted from the addition of a new hyperlink, or the flip (change of sign) of a hyperlink. We could also reduce the computational time by considering the number of triads of each type that are created when an edge is added (or flipped).

## Proposed timeline


## Organization within the team
*A list of internal milestones up until project milestone P4. Add here a sketch of your planning for the next project milestone.*
## Questions for TAs (optional)
*Add here any questions you have for us related to the proposed project.*
