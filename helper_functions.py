#%%
import pandas as pd
import numpy as np
import random
import pickle 
from tqdm import tqdm
from scipy.sparse import csr_matrix,find
from scipy import linalg
import networkx as nx
import itertools
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from openTSNE import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import mixture
import matplotlib as mpl
# import extension_new_functions
import statsmodels.api as sm
import seaborn as sns

#%% Clustering : plots
def plot_simple(labels, c_labels, embedding_reduced):
    plt.figure(figsize=(12, 12))    
    scatter = plt.scatter(embedding_reduced[:,0], embedding_reduced[:,1],
        alpha=0.6,s=0.4, c=c_labels, cmap='tab20',)
    plt.title('Selected GMM: full model, {} components'.format(len(np.unique(labels))))
    lgnd = plt.legend(handles=scatter.legend_elements(num=None)[0], 
                labels=labels,
                markerscale=2, loc="upper left", scatterpoints=1, fontsize=10,
                ncol=int(len(np.unique(labels))/2)) 
    plt.show()


def plot_GMM(gmm, X):
    plt.figure(figsize=(8, 6))
    cm = plt.cm.get_cmap('tab20')
    splot = plt.subplot(1, 1, 1)
    Y_ = gmm.predict(X)
    for i, (mean, cov) in enumerate(zip(gmm.means_, gmm.covariances_)):
        v, w = linalg.eigh(cov)
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8,label=i,color=cm.colors[i])

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180. * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(.5)
        splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())
    plt.title('Selected GMM: full model, {} components'.format(len(gmm.means_)))
    plt.subplots_adjust(hspace=.35, bottom=.02)
    lgnd = plt.legend(loc="lower left", scatterpoints=1, fontsize=10,) #ncol=n_components)
    for handle in lgnd.legendHandles:
        handle.set_sizes([12.0])
    plt.show()

#%% Clustering : general
def dim_reduction(embedding, which, dim_out, dim_in, t_sne_params, labels_= None):
    """
    Reduces the dimension of embedding and plots the identified clusters if labels are given 
    """
    print("Dimension reduction with {}".format(which))
    if which == "PCA":
         embedding_reduced = PCA(n_components=dim_out).fit(embedding).transform(embedding)

    elif which == "t-SNE":
        if dim_in >0:
            embedding = PCA(n_components=dim_in,).fit(embedding).transform(embedding)
            print("PCA Reduction done")
        else:
            embedding= np.array(embedding)
        perplexity, exaggeration, n_iter, early_exaggeration,early_exaggeration_iter,                                   learning_rate = t_sne_params
        model=TSNE(n_components=2, perplexity=perplexity, random_state=0,
                                verbose=True,exaggeration=exaggeration,n_iter=n_iter,
                                early_exaggeration=early_exaggeration,                                                              early_exaggeration_iter=early_exaggeration_iter,
                                learning_rate=learning_rate)
        embedding_reduced = model.fit(embedding)
    else: print("Sorry, reduction algorithm {} doesn't exist".format(which))

    #if labels for clusters are given plot the clusters
    if labels_ is not None:
        plot_simple(labels, embedding_reduced)
    return embedding_reduced


def cluster_number_analysis(X, n=11):
    ##Silhouettes
    silhouettes = []
    sse = []


    # Try multiple k
    for k in range(2, n):
        # Cluster the data and assigne the labels
        model = KMeans(n_clusters=k, random_state=10)
        labels = model.fit_predict(X)
        print("KMeans done, ", k)
        # Get the Silhouette score
        score = silhouette_score(X, labels)
        silhouettes.append({"k": k, "score": score})
        sse.append({"k": k, "sse": model.inertia_})
        
    # Convert to dataframe
    silhouettes = pd.DataFrame(silhouettes)

    # Plot the data
    fig, axs = plt.subplots(1, 1, figsize=(7,7), sharey=True)
    axs.plot(silhouettes.k, silhouettes.score)
    axs.set_xlabel("K")
    axs.set_ylabel("Silhouette score")

    ##Elbow

    sse = pd.DataFrame(sse)
    # Plot the data
    fig, axs = plt.subplots(1, 1, figsize=(7,7), sharey=True)
    axs.plot(sse.k, sse.sse)
    axs.set_xlabel("K")
    axs.set_ylabel("Sum of Squared Errors")


def categories(labels, embedding, n_first=40):
    """
    Prints all categories discoverd by the clustering algorithm
    """
    for i in range(len(np.unique(labels))):
        print(i)
        print(np.array(embedding.index)[labels==i][:n_first])


# ----------------------------------------------------------------------------------
# The following functions were already presented in the notebook.  


tqdm.pandas()
columns_status = ["t"+str(i+1) for i in range(16)]
columns_balance = ["t3B", "t0B", "t12B"]
columns_balance_full = ["t3B", "t1B", "t2B", "t0B"]

def same_value(s):
    """Check if an array has an unique value"""
    a = s.to_numpy()
    return (a[0] == a).all()


def selectLastEdge(df):
    # use numpy array as concat is much faster-> then we explode the dataframe to have all of them
    # if only one observation
    if df.shape[0] == 1:
        return df.head(1).id.to_numpy()
    # if all values are the same
    if same_value(df.LINK_SENTIMENT):
        return df.head(1).id.to_numpy()
    # +- or -+ (signs are different)
    if df.shape[0] == 2:
        return df.tail(1).id.to_numpy()

    # othewise we compute the median delta time (-> estimates the freq. at which edges appear)
    median_deltatime = (df.TIMESTAMP.shift(-1)-df.TIMESTAMP).median()
    start, end = (df.TIMESTAMP.tail(1)-median_deltatime).item(), df.TIMESTAMP.tail(1).item()
    nb_signs = df.query(
        '@start <= TIMESTAMP <= @end').LINK_SENTIMENT.value_counts()
    # if we have a tie, we look what is the dominant sign in the last observation
    if len(nb_signs) > 1 and same_value(nb_signs):
        # moving avg from the end (consider only odd positions)
        # e.g. ++--+- -> -1,0,-1,-2,... -> 0,-2
        sign_avg = df.LINK_SENTIMENT.sort_index(ascending=False).cumsum()[1::2]
        if all(sign_avg == 0):  # +-+-+-+-+- or -+-+-+-+
            return df.tail(1).id.to_numpy()
        else:
            # search for the first negative or positive negative occurrence in sign_avg
            # then if this is positive assign -> +1 otherwise -> -1
            sign = 1 if sign_avg[(sign_avg != 0).idxmax()] > 0 else -1
    else:
        # take the dominant sign in the time interval
        sign = nb_signs.idxmax()

    # find the last group with the corresponding sign
    # here we first take the first occurrence of each group and sort them in decreasing order (wrt. the dates)
    firstOccurenceReverse = df.groupby([(df.LINK_SENTIMENT != df.LINK_SENTIMENT.shift(
    )).cumsum()]).head(1).sort_index(ascending=False)
    # the we filter by the signs and take the first occurrence
    return firstOccurenceReverse[firstOccurenceReverse.LINK_SENTIMENT == sign].head(1).id.to_numpy()


# %%

def baseline(df):
    """Compute generative and receptive baseline"""
    gen = ((df.groupby(by="SOURCE_SUBREDDIT").mean()+1)/2.0).squeeze()
    rec = ((df.groupby(by="TARGET_SUBREDDIT").mean()+1)/2.0).squeeze()
    outdegree = df.groupby(by="SOURCE_SUBREDDIT").apply(len)
    indegree = df.groupby(by="TARGET_SUBREDDIT").apply(len)
    return gen.to_dict(), rec.to_dict(), outdegree.to_dict(), indegree.to_dict()

# %%
# open issue https://github.com/scipy/scipy/issues/12495
def convert_to_64bit_indices(A):
    A.indptr = np.array(A.indptr, copy=False, dtype=np.int64)
    A.indices = np.array(A.indices, copy=False, dtype=np.int64)
    return A


def compute_trace(A, B, C):
    """Compute tr(A*B*C) where A,B,C are sparse matrices"""
    # function defined in the first part
    A2 = convert_to_64bit_indices(A.dot(B))
    # equivalent to tr(A*B*C) but much faster...
    return A2.multiply(C.transpose()).sum()


def createGraph(df):
    """create a networkx oriented graph from a pandas dataframe"""
    G = nx.from_pandas_edgelist(df, source='SOURCE_SUBREDDIT',
                                target='TARGET_SUBREDDIT', edge_attr=None, create_using=nx.DiGraph())
    return G


def total_triad(df):
    """Count the total number of triad"""
    G = createGraph(df)
    B = nx.adjacency_matrix(G)
    A = B+B.T
    return int(compute_trace(A, A, A)/6)

# total_triad(last_network)


# %%
def get_one_hot(targets, nb_classes):
    """Convert targerts to one-hot encoding"""
    res = np.eye(nb_classes, dtype=np.int32)[
        np.array(targets, dtype=np.int32).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


def common_neighbors(G, u, v):
    """Find nodes that are linked to u and v in the graph G"""
    return (w for w in set(G.predecessors(u)).union(set(G.successors(u))) if w in set(G.predecessors(v)).union(set(G.successors(v))) and w not in (u, v))


def buildCovariate_quadrant(quadrant, wAC, wBC, *args, **kwargs):
    """
     1  2 |  3  4 
     5  6 |  7  8 
     ------------
     9 10 | 11 12
    13 14 | 15 16
    update one of the 4 quadrants

    parameters:
    quadrant: either 1/3/9/11
    wAC: weight of AC (or CA) edge
    wBC: weight of BC (or CB) edge

    returns the features of the given triad (e.g. nodes, type (t1,t2,...), indegree of B, outdegree of A)
    """
    if wAC == 1:
        if wBC == 1:
            case = quadrant
            return buildCovariate_status(case=case, *args, **kwargs)
        else:
            case = quadrant+1
            return buildCovariate_status(case=case, *args, **kwargs)
    else:
        if wBC == 1:
            case = quadrant+4
            return buildCovariate_status(case=case, *args, **kwargs)
        else:
            case = quadrant+5
            return buildCovariate_status(case=case, *args, **kwargs)


def buildCovariate_status(case, wAB, outdegree, indegree, gen_baseline, rec_baseline, nodeA, nodeB, nodeC, timestamp, **kwargs):
    return ( [wAB, outdegree, indegree, gen_baseline,
            rec_baseline, nodeA, nodeB, nodeC, timestamp] + 
            get_one_hot(np.array([case-1]), 16).squeeze().tolist() )


def buildDataset(df, baselines_gen, baselines_rec, outdegree, indegree):
    """
    returns:
    large data set of triads with these columns:
    "y", "outdegree", "indegree", "generative_baseline",
    "receptive_baseline", "nodeA", "nodeB", "nodeC",
    "timestamp"
    """
    # create the graph and estimate the total number of triad to pre-allocate memory.
    # also checks that the number of enumerate triads is the correct one
    G = createGraph(df)
    outdegrees = nx.out_degree_centrality(G)
    # faster to collect the results in a numpy array (we will then convert it to a dataframe)
    data = np.zeros((total_triad(df), 25))

    # create directed graph
    uniqueNode = df.SOURCE_SUBREDDIT.append(df.TARGET_SUBREDDIT).unique()
    index = {node: i for i, node in enumerate(uniqueNode)}
    # create a dictionary mapping an index to each subreddit(node) 
    node_dictionary = {i: node for i, node in enumerate(uniqueNode)}
    # create an empty graph. We will add iteratively an edge to it and count the triad that are created
    G = nx.DiGraph()
    G.add_nodes_from(uniqueNode)
    i = 0
    # iteration over the edges
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        G.add_edge(row.SOURCE_SUBREDDIT, row.TARGET_SUBREDDIT,
                   weight=row.LINK_SENTIMENT)
        A, B = row.SOURCE_SUBREDDIT, row.TARGET_SUBREDDIT
        wAB = row.LINK_SENTIMENT
        baseline_node_A = baselines_gen[A]
        baseline_node_B = baselines_rec[B]

        outdegreeA = outdegree[A] if outdegree is not None else 1
        indegreeB = indegree[B] if indegree is not None else 1
        params = {"wAB": (wAB + 1)/2, "gen_baseline": baseline_node_A,
                "rec_baseline": baseline_node_B, "outdegree": outdegreeA,
                "indegree": indegreeB, "nodeA": index[A], "nodeB": index[B],
                "timestamp": row.TIMESTAMP.to_numpy()}
        # iteration over the nodes that are linked to A and B
        # form a contextualised link (edge AB come after AC and BC)
        for C in common_neighbors(G, row.SOURCE_SUBREDDIT, row.TARGET_SUBREDDIT):
            # translate to integer
            params["nodeC"] = index[C]
            # add the triad to the dataset
            if G.has_edge(A, C):
                if G.has_edge(B, C):
                    data[i] = buildCovariate_quadrant(quadrant=3, wAC=G.get_edge_data(
                        A, C)["weight"], wBC=G.get_edge_data(B, C)["weight"], **params)
                    i += 1
                if G.has_edge(C, B):
                    data[i] = buildCovariate_quadrant(quadrant=1, wAC=G.get_edge_data(
                        A, C)["weight"], wBC=G.get_edge_data(C, B)["weight"], **params)
                    i += 1
            if G.has_edge(C, A):
                if G.has_edge(B, C):
                    data[i] = buildCovariate_quadrant(quadrant=11, wAC=G.get_edge_data(
                        C, A)["weight"], wBC=G.get_edge_data(B, C)["weight"], **params)
                    i += 1
                if G.has_edge(C, B):
                    data[i] = buildCovariate_quadrant(quadrant=9, wAC=G.get_edge_data(
                        C, A)["weight"], wBC=G.get_edge_data(C, B)["weight"], **params)
                    i += 1
    # convert to a dataframe 
    dataSet = pd.DataFrame(data, columns=["y", "outdegree", "indegree", "generative_baseline",
                                        "receptive_baseline", "nodeA", "nodeB", "nodeC",
                                        "timestamp"]+["t"+str(i+1) for i in range(16)])
    dataSet.timestamp = pd.to_datetime(dataSet.timestamp)
    cols = ["nodeA", "nodeB", "nodeC"]

    # translate the integer to a string (we could not store strings in the numpy array)
    def translateIndex(row):
        return node_dictionary[row]
    
    dataSet[cols] = dataSet[cols].applymap(translateIndex)
    return dataSet.sort_values("timestamp")


# %%

# classify the categories from status to the corresponding ones in balance
def buildBalanceCategory(dataset):
    # Let _/_/_ be the sign of the corresponding edges AB/BC/AC

    # correspond to ?/+/+
    dataset["t3B"] = dataset.t1+dataset.t3+dataset.t9+dataset.t11
    # correspond to ?/+/-
    dataset["t2B"] = dataset.t5+dataset.t7+dataset.t13+dataset.t15
    # correspond to ?/-/+
    dataset["t1B"] = dataset.t2+dataset.t4+dataset.t10+dataset.t12
    dataset["t12B"] = dataset["t1B"]+dataset["t2B"]
    # correspond to ?/-/-
    dataset["t0B"] = dataset.t6+dataset.t8+dataset.t14+dataset.t16

# %%

def compute_triad(dataset, normalisation=False):
    # classify the triads using the sign of the last edge (AB)
    # (the sign of AB belongs to {0,1})
    # t3B + AB=1 => +++
    # t2B + AB=0 or t0B + AB=1 or t1B + AB=0 => +--
    # t2B + AB=1 or t3B + AB=0 or t1B + AB=1 => ++-
    # t0B + AB=0 => ---
    balance_columns = dataset[columns_balance_full]
    if normalisation:
        balance_columns = balance_columns/dataset.outdegree[:, None]
    # multiply the indicator columns be the sign of AB, then sum over all the rows 
    # (we obtain the number of triads in each category)
    separated_counts = pd.concat((balance_columns*dataset.y[:, None], balance_columns*(
        1-dataset.y[:, None])), axis=1).sum(axis=0).reset_index(drop=True)
    retVal = []
    # combine the columns using the rules defined above
    for l in [[0], [3, 5, 6], [1, 2, 4], [7]]:
        tot_per_cat = 0
        for i in l:
            tot_per_cat += separated_counts[i]
        retVal.append(tot_per_cat)
    total = separated_counts.sum()
    # convert to a pandas dataframe
    table = pd.Series(retVal).to_frame("Triad")
    table = table.rename(index={0: '+++', 1: "+--", 2: "++-", 3: "---"})
    table["p(T)"] = table.Triad/total
    return table, total


def surprise_balance(table, total):
    """Compute the surprise as defined in the article"""
    table["s(T)"] = (table["Triad"]-total*table["p_0(T)"]) / \
        np.sqrt(total*table["p_0(T)"]*(1-table["p_0(T)"]))
    return table


def computeTable3(dataset, dataset_shuffled, normalisation=False):
    """Reproduce table 3 of the article"""
    if not normalisation:
        # compute the number of triads of each type and their fraction
        table, total = compute_triad(dataset, normalisation)
        # check that the sum is equal to the number of triads of the first part
        print("Total Number of triad : {:.0f}".format(total))
        # shuffle the signs and compute the number of triads of each type
        table["p_0(T)"] = compute_triad(dataset_shuffled, False)[0]["p(T)"]
        # compute the surprise
        table = surprise_balance(table, total)
    else:
        table = compute_triad(dataset, True)[0]["Triad"].to_frame("Weighted_Triad")
        table["Weighted_Triad_shuffled"] = compute_triad(
            dataset_shuffled, True)[0]["Triad"]
        table["diff"] = table.Weighted_Triad-table.Weighted_Triad_shuffled
    return table




# %%


def surprise_status(table, norm):
    """Compute generative surprise and receptive surprise"""
    table["sg"] = (table.plus_countG-table.gb)/np.sqrt(table.gb *
                                                       (1-table.gb/table["count"])) if norm else (table.plus_countG-table.gb)
    table["sr"] = (table.plus_countR-table.rb)/np.sqrt(table.rb *
                                                       (1-table.rb/table["count"])) if norm else (table.plus_countR-table.gb)




def status_count(dataset, normalisation=False):
    """Count the number of triads in each category (t1,t2,t3,....)
    if normalisation is true: normalise the counts using the outdegree/indegree
    """
    # initialisation
    counts = dataset[columns_status].sum()

    generative_baseline = dataset["generative_baseline"]
    receptive_baseline = dataset["receptive_baseline"]
    if normalisation:
        generative_baseline = generative_baseline/dataset["outdegree"]
        receptive_baseline = receptive_baseline/dataset["indegree"]
    # use the indicators in columns_status to assign the generative baseline to each triad and sum them
    # -> find the generative baseline associated to each type of triad
    generative_baseline = (
        dataset[columns_status]*generative_baseline[:, None]).sum()
    # use the indicators in columns_status to assign the receptive baseline to each triad and sum them
    # -> find the receptive baseline associated to each type of triad
    receptive_baseline = (dataset[columns_status]
                          * receptive_baseline[:, None]).sum()
    # find triad with positive edge AB
    positive = dataset[dataset.y == 1]
    if normalisation:
        plusG = (positive[columns_status]/positive["outdegree"][:, None]).sum()
        plusR = (positive[columns_status]/positive["indegree"][:, None]).sum()
    else:
        plusG = positive[columns_status].sum()
        plusR = plusG
    return counts, plusG, plusR, generative_baseline, receptive_baseline


# %%


def plotComparison(df1, df2, which, on, typePlot="dens"):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharey=True)

    def plotHist(val, ax, bins, labels, xlab, title):
        ax.hist(val, bins=bins, label=labels, density=True, rwidth=0.9)
        ax.legend(loc='upper right')
        ax.set_xlabel(xlab)
        ax.set_ylabel('Distribution')
        ax.set_title(title)

    def plotDens(vals, ax, bins, labels, xlab, title, useLogx=True):
        for val, label in zip(vals, labels):
            val += 1e-8
            sns.kdeplot(val, shade=True, linewidth=3,
                        label=label, ax=ax, log_scale=useLogx)
        ax.legend(loc='upper right')
        ax.set_xlabel(xlab)
        ax.set_ylabel('Distribution')
        ax.set_title(title)
    titles = ["Actual network", "Shuffled network"]
    for df, ax, title in zip([df1, df2], axs, titles):
        
        value1 = df[df.LINK_SENTIMENT == -1].score# _x+df[df.LINK_SENTIMENT==-1].score_y)/2
       
        value2 = df[df.LINK_SENTIMENT == 1].score # _x+df[df.LINK_SENTIMENT==1].score_y)/2
        xlabel = ""
        if which == "out":
            xlabel += "Outdegree"
        elif which == "in":
            xlabel += "Indegree"
        xlabel += " of the "+on+" node (log)"
        if typePlot == "dens":
            plotDens([value1, value2], ax, 20, [
                     "Negative", "Positive"], xlabel, title)
        else:
            plotHist([value1, value2], ax, 20, [
                     "Negative", "Positive"], xlabel, title)
    fig.tight_layout()
    plt.show()

def centralityFunc(df, which, on):
    """Compare the centrality score of edges with negative values and the ones with positive values
    which: either out/in for outdegree/indegree
    on: either source/target. Choose which node of the edge should be compared.
    """
    G = createGraph(df)
    # compute centrality scores
    if which == "out":
        centrality_score = pd.Series(nx.out_degree_centrality(
            G)).to_frame().rename(columns={0: "score"})
    elif which == "in":
        centrality_score = pd.Series(nx.in_degree_centrality(
            G)).to_frame().rename(columns={0: "score"})
    else:
        print("unkown command")
        return
    df_ = df.copy()
    if on == "source":
        # merge with source
        df_ = df_.merge(centrality_score,
                        left_on="SOURCE_SUBREDDIT", right_index=True)
    elif on == "target":
        # merge with target
        df_ = df_.merge(centrality_score,
                        left_on="TARGET_SUBREDDIT", right_index=True)
    else:
        print("unkown command")
        return
    return df_


def CompareCentrality(df1, df2, which, on):
    """Compare the centrality of edges with negative values and the ones with positive values in df1 and df2"""
    df1_ = centralityFunc(df1, which, on)
    df2_ = centralityFunc(df2, which, on)
    plotComparison(df1_, df2_, which, on)




# %%

def consistency(table):
    """ Check if balance/status are consistent with the receptive/generative surprise """
    A_status = np.ones(16)
    A_status[[0, 1, 2, 3, 12, 13, 14, 15]] = -1
    B_status = np.ones(16)
    B_status[[1, 2, 5, 6, 9, 10, 13, 14]] = -1
    balance = np.ones(16)
    balance[[1, 3, 4, 6, 9, 11, 12, 14]] = -1
    table["Sg"] = (table.sg * B_status) >= 0
    table['Sr'] = (table.sr * A_status) < 0
    table['Bg'] = (table.sg * balance) >= 0
    table['Br'] = (table.sr * balance) >= 0
