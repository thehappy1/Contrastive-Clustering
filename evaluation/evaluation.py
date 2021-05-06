import numpy as np
from sklearn import metrics
from munkres import Munkres
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate(label, pred, extracted_features, dataset):
    print(pred[:5000])
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    f = metrics.fowlkes_mallows_score(label, pred)
    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    acc = metrics.accuracy_score(pred_adjusted, label)
    ds = metrics.davies_bouldin_score(extracted_features, pred)
    s = metrics.silhouette_score(extracted_features, pred, metric='euclidean')
    from s_dbw import S_Dbw
    s_dbw = S_Dbw(extracted_features, pred)
    compute_tsne(features=extracted_features, label=label, dataset=dataset)
    return nmi, ari, f, acc, ds, s, s_dbw

def compute_tsne(features, label, dataset):
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, perplexity=20, n_jobs=16, random_state=0, verbose=0).fit_transform(features)

    viz_df = pd.DataFrame(data=tsne[:5000])
    viz_df['Label'] = label[:5000]

    if dataset == "FASHION-MNIST":
        dict = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt",
                7: "Sneaker", 8: "Bag",
                9: "Ankle boot"}
    else:
        dict = {0: "Shirts", 1: "Watches", 2: "T-Shirts", 3: "C. Shoes", 4: "Handbags", 5: "Tops", 6: "Kurtas",
                 7: "S. Shoes", 8: "Heels", 9 : "Sunglasses"}

    viz_df['Label'] = viz_df["Label"].map(dict)

    print("test 1: ", viz_df.Label.tolist())
    print("test 2:", viz_df['Label'].unique())

    viz_df.to_csv('tsne.csv')
    plt.subplots(figsize=(14, 7))
    sns.scatterplot(x=0, y=1, hue=viz_df.Label.tolist(), legend='full', hue_order=sorted(viz_df['Label'].unique()),
                    palette=sns.color_palette("hls", n_colors=10),
                    alpha=.5,
                    data=viz_df)
    l = plt.legend(bbox_to_anchor=(-.1, 1.00, 1.1, .5), loc="lower left", markerfirst=True,
                   mode="expand", borderaxespad=0, ncol=10 + 1, handletextpad=0.01, )

    l.texts[0].set_text("")
    plt.ylabel("")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(dataset+'_tnse.png', dpi=150)
    plt.clf()


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred
