import numpy as np
import math

def KMax(arr, k):
    # Sort the given array arr in reverse
    # order.
    arr.sort(reverse = True)
    # Print the first kth largest elements
    _kmax = []
    for i in range(min(k,arr.__len__())):
        _kmax.append(arr[i])
    return  np.mean(_kmax);


def KMean(arr, k):
    total = 0
    n = arr.__len__()
    # base case if 2*k>=n
    # means all element get removed
    if (2 * k >= n):
        return np.mean(arr)

    # first sort all elements
    arr.sort()

    start, end = k, n - k - 1

    # sum of req number
    for i in range(start, end + 1):
        total += arr[i]

        # find average
    return (total / (n - 2 * k))

class RepMethod():
    def __init__(self,
                 align_info=None,
                 p=None,
                 k=10,
                 add_cluster_feature=False,
                 max_layer=None,
                 feature_length=10,
                 featurename='histon',
                 featuremethod='fast',
                 frameworks=['regal'],
                 alpha=0.1,
                 num_buckets=None,
                 min_bucket_length=1,
                 normalize=False,
                 gammastruc=1,
                 gammaattr=1):
        self.add_cluster_feature = add_cluster_feature
        self.feature_length = feature_length
        self.featuremethod = featuremethod
        self.featurename = featurename
        self.frameworks = frameworks
        self.p = p  # sample p points
        self.k = k  # control sample size
        self.max_layer = max_layer  # furthest hop distance up to which to compare neighbors
        self.alpha = alpha  # discount factor for higher layers
        self.num_buckets = num_buckets  # number of buckets to split node feature values into #CURRENTLY BASE OF LOG SCALE
        self.min_bucket_length = min_bucket_length
        self.normalize = normalize  # whether to normalize node embeddings
        self.gammastruc = gammastruc  # parameter weighing structural similarity in node identity
        self.gammaattr = gammaattr  # parameter weighing attribute similarity in node identity


class Graph():
    # Undirected, unweighted
    def __init__(self,
                 adj,
                 add_cluster_feature=False,
                 min_bucket_length=1,
                 num_buckets=None,
                 node_labels=None,
                 edge_labels=None,
                 graph_label=None,
                 node_attributes=None,
                 graph=None,
                 true_alignments=None):

        self.add_cluster_feature = add_cluster_feature
        self.graph = graph;
        self.G_adj = adj  # adjacency matrix
        self.N = self.G_adj.shape[0]  # number of nodes

        #self.node_degrees = np.ravel(np.sum(self.G_adj, axis=0).astype(int))
        self.node_degrees = np.zeros([self.N]);
        for (x,y) in self.graph.degree:
            self.node_degrees[x] = y;
        np.ravel(np.sum(self.G_adj, axis=0).astype(int))

        self.node_grarid = np.zeros([self.N])

        max_exp = 1
        min_exp = 1
        for node in range(self.N):
            ns = np.nonzero(self.G_adj[node])[-1].tolist();
            ns.append(node);
            self.node_grarid[node] =  (math.log(self.node_degrees[node]+1))/ \
                                  (np.mean([math.log(self.node_degrees[nn]+1) for nn in ns])+1);

        max_exp = 1
        min_exp = 1
        # component number

        self.component_index = [int(0)] * self.N
        self.component_number_0 = 0
        bfs = [0] * self.N;
        current_component_index = int(0)

        f_ind = 0
        while (True):
            while f_ind < self.N:
                if (bfs[f_ind] == 0):
                    source = f_ind;
                    break;
                f_ind += 1

            if (f_ind >= self.N):
                break;

            f_ind = f_ind + 1
            if (self.component_number_0 == 0 and f_ind > self.N / 2):
                self.component_number_0 = current_component_index

            ns = [source]
            self.component_index[source] = current_component_index
            bfs[source] = 1

            while ns.__len__() > 0:
                n = ns.pop(0)
                for nn in graph.neighbors(n):
                    if (bfs[nn] == 0):
                        ns.append(nn);
                        bfs[nn] = 1
                        self.component_index[nn] = current_component_index
            current_component_index += 1
        #
        self.tempgraph = self.graph.copy();
        self.component_number = current_component_index
        self.component_index_d = np.zeros([self.component_number]);
        for t in range(self.N):
            self.component_index_d[self.component_index[t]] +=1
        self.max_degree = max(self.node_degrees)
        self.num_buckets = num_buckets  # how many buckets to break node features into
        self.min_bucket_length = min_bucket_length
        self.node_labels = node_labels
        self.edge_labels = edge_labels
        self.graph_label = graph_label
        self.node_attributes = node_attributes  # N x A matrix, where N is # of nodes, and A is # of attributes
        self.kneighbors = None  # dict of k-hop neighbors for each node
        self.true_alignments = true_alignments  # dict of true alignments, if this graph is a combination of multiple graphs
    def normalize(self,x ):
        return np.exp(-.5*np.square((x-np.mean(x))/(np.std(x))))

    def agg_relative(self,x,num=3,max_exp = 1 , min_exp = 1 , normalization = False):
        temp = np.zeros([self.N])
        for k in range(num):
            if(normalization == True):
                x = self.normalize(x) + 1
            for node in range(self.N):
                ns = np.nonzero(self.G_adj[node])[-1].tolist();
                ns.append(node);
                temp[node] = (math.pow(x[node],max_exp)) / (np.mean([math.pow(x[nn],min_exp)  for nn in ns] ));

            x = temp;
        return x

    def agg_mean(self,x,num=3):
        temp = np.zeros([self.N])
        for k in range(num):

            for node in range(self.N):
                ns = np.nonzero(self.G_adj[node])[-1].tolist();
                ns.append(node);
                temp[node] = np.mean([x[nn]  for nn in ns] )

            x = temp;
        return x