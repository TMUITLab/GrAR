from tkinter.tix import _dummyFrame

import numpy as np, scipy as sp, networkx as nx
import scipy.sparse as sp

import math, time, os, sys
from sklearn.neighbors import KDTree
import collections
from sklearn.preprocessing import normalize
import config as cnf
import statistics
import matplotlib.pyplot as plt
import alignments
from config import *
import random
import matplotlib.pyplot as plt

from ge.classify import read_node_label, Classifier
from ge import *

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE


def make_orthonormal_matrix(n):
    """
    Makes a square matrix which is orthonormal by concatenating
    random Householder transformations
    """
    A = np.identity(n)
    d = np.zeros(n)
    d[n-1] = random.choice([-1.0, 1.0])
    for k in range(n-2, -1, -1):
        # generate random Householder transformation
        x = np.random.randn(n-k)
        s = math.sqrt((x**2).sum()) # norm(x)
        sign = math.copysign(1.0, x[0])
        s *= sign
        d[k] = -sign
        x[0] += s
        beta = s * x[0]
        # apply the transformation
        y = np.dot(x,A[k:n,:]) / beta
        A[k:n,:] -= np.outer(x,y)
    # change sign of rows
    A *= d.reshape(n,1)
    return A

def plotgraph(G):
	degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
	# print "Degree sequence", degree_sequence
	degreeCount = collections.Counter(degree_sequence)
	deg, cnt = zip(*degreeCount.items())

	fig, ax = plt.subplots()
	plt.bar(deg, cnt, width=0.80, color='b')

	plt.title("Degree Histogram")
	plt.ylabel("Count")
	plt.xlabel("Degree")
	ax.set_xticks([d + 0.4 for d in deg])
	ax.set_xticklabels(deg)

	# draw graph in inset
	plt.axes([0.4, 0.4, 0.5, 0.5])
	Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
	pos = nx.spring_layout(G)
	plt.axis('off')
	nx.draw_networkx_nodes(G, pos, node_size=20)
	nx.draw_networkx_edges(G, pos, alpha=0.4)

	plt.show()

#Input: graph, RepMethod
#Output: dictionary of dictionaries: for each node, dictionary containing {node : {layer_num : [list of neighbors]}}
#        dictionary {node ID: degree}
def get_khop_neighbors(graph, rep_method):
	if rep_method.max_layer is None:
		rep_method.max_layer = graph.N #Don't need this line, just sanity prevent infinite loop

	kneighbors_dict = {}

	#only 0-hop neighbor of a node is itself
	#neighbors of a node have nonzero connections to it in adj matrix
	for node in range(graph.N):
		neighbors = np.nonzero(graph.G_adj[node])[-1].tolist() ###
		if len(neighbors) == 0: #disconnected node
			print("Warning: node %d is disconnected" % node)
			kneighbors_dict[node] = {0: set([node]), 1: set()}
		else:
			if type(neighbors[0]) is list:
				neighbors = neighbors[0] 
			kneighbors_dict[node] = {0: set([node]), 1: set(neighbors) - set([node]) } 

	#For each node, keep track of neighbors we've already seen
	all_neighbors = {}
	for node in range(graph.N):
		all_neighbors[node] = set([node])
		all_neighbors[node] = all_neighbors[node].union(kneighbors_dict[node][1])

	#Recursively compute neighbors in k
	#Neighbors of k-1 hop neighbors, unless we've already seen them before
	current_layer = 2 #need to at least consider neighbors
	max_layer = rep_method.max_layer;
	if(rep_method.max_layer == None or rep_method.max_layer >= 2):
		max_layer = 2

	while True:
		if current_layer > max_layer: break
		reached_max_layer = True #whether we've reached the graph diameter

		for i in range(graph.N):
			#All neighbors k-1 hops away
			neighbors_prevhop = kneighbors_dict[i][current_layer - 1]
			
			khop_neighbors = set()
			#Add neighbors of each k-1 hop neighbors
			for n in neighbors_prevhop:
				neighbors_of_n = kneighbors_dict[n][1]
				for neighbor2nd in neighbors_of_n: 
					khop_neighbors.add(neighbor2nd)

			#Correction step: remove already seen nodes (k-hop neighbors reachable at shorter hop distance)
			khop_neighbors = khop_neighbors - all_neighbors[i]

			#Add neighbors at this hop to set of nodes we've already seen
			num_nodes_seen_before = len(all_neighbors[i])
			all_neighbors[i] = all_neighbors[i].union(khop_neighbors)
			num_nodes_seen_after = len(all_neighbors[i])

			#See if we've added any more neighbors
			#If so, we may not have reached the max layer: we have to see if these nodes have neighbors
			if len(khop_neighbors) > 0:
				reached_max_layer = False 

			#add neighbors
			kneighbors_dict[i][current_layer] = khop_neighbors #k-hop neighbors must be at least k hops away

		if reached_max_layer:
			break #finished finding neighborhoods (to the depth that we want)
		else:
			current_layer += 1 #move out to next layer

	return kneighbors_dict

def get_degree_coef (graph,rep_method):
	if rep_method.num_buckets is not None:
		maxdim = int(math.log(graph.max_degree/rep_method.min_bucket_length, rep_method.num_buckets) + 1);
		bins = np.zeros([maxdim, 2], dtype=float);

		for node in range(graph.N):
			degree = graph.node_degrees[node];
			ind = int(math.log(degree/rep_method.min_bucket_length, rep_method.num_buckets))
			bins[ind,int(node/(graph.N/2))] += 1;
	else:

		bins = np.zeros([(int(graph.max_degree/rep_method.min_bucket_length) + 1), 2], dtype=float);

		for node in range(graph.N):
			degree = graph.node_degrees[node];
			bins[int(degree/rep_method.min_bucket_length)] += 1;

	coef = np.zeros(bins.shape[0], dtype=float);
	sum = bins[:,0]-bins[:,1];

	for i in range(bins.shape[0]-1).__reversed__():
		sum[i] += sum[i+1]

	for i in range(bins.shape[0]).__reversed__():
		if(bins[i,0] >0):
			coef [i] = sum[i]/bins[i,0]
		else:
			coef[i] = 0

	return  coef;

def get_degree_x_coef (graph,rep_method):
	if rep_method.num_buckets is not None:
		maxdim = int(math.log(graph.max_degree/rep_method.min_bucket_length, rep_method.num_buckets) + 1);
		bins = np.zeros([maxdim, 2], dtype=float);

		for node in range(graph.N):
			degree = graph.node_degrees[node];
			v = math.log(degree/rep_method.min_bucket_length, rep_method.num_buckets)

			bin = int(v);
			x = math.fabs(bin + 0.5 - v);
			bins[int(v)] += 1 - x;

			if (bin + 0.5 < v and bin < maxdim):
				bins[bin + 1] += x
			elif (bin + 0.5 > v and bin > 0):
				bins[bin - 1] += x

			bins[bin,int(node/(graph.N/2))] += graph.node_degrees[node];

	else:

		bins = np.zeros([(int(graph.max_degree/rep_method.min_bucket_length) + 1), 2], dtype=float);

		for node in range(graph.N):
			degree = graph.node_degrees[node];
			bins[int(degree / rep_method.min_bucket_length)] += graph.node_degrees[node]

	coef = np.zeros([bins.shape[0]], dtype=float);
	for i in range(bins.shape[0]).__reversed__():
		if(bins[i,0] > 0):
			coef [i] = bins[i,1]/bins[i,0]
		else:
			coef[i] = 1

	return  coef;

#Turn lists of neighbors into a degree sequence
#Input: graph, RepMethod, node's neighbors at a given layer, the node
#Output: length-D list of ints (counts of nodes of each degree), where D is max degree in graph
def get_degree_sequence(graph, rep_method, kneighbors, current_node):
	if rep_method.num_buckets is not None:
		degree_counts = [0] * int(math.log(graph.max_degree/rep_method.min_bucket_length, rep_method.num_buckets) + 1)
	else:
		degree_counts = [0] * (int(graph.max_degree/rep_method.min_bucket_length) + 1)

	#For each node in k-hop neighbors, count its degree
	for kn in kneighbors:
		weight = 1 #unweighted graphs supported here
		degree = graph.node_degrees[kn]
		if rep_method.num_buckets is not None:
			try:
				v = math.log(degree/rep_method.min_bucket_length, rep_method.num_buckets);
				#if(int(v) > 0):
				#	degree_counts[int(v)-1] += weight/2
				degree_counts[int(v)] += weight

			except:
				print("Node %d has degree %d and will not contribute to feature distribution" % (kn, degree))
		else:
			degree_counts[int(degree/rep_method.min_bucket_length)] += weight


	return degree_counts/graph.node_degrees[current_node];

#Turn lists of neighbors into a dtsolregree sequence
#Input: graph, RepMethod, node's neighbors at a given layer, the node
#Output: length-D list of ints (counts of nodes of each degree), where D is max degree in graph
def get_degree_sequence_histon(graph, rep_method, kneighbors, current_node):
	if rep_method.num_buckets is not None:
		degree_counts = np.zeros([int(math.log(graph.max_degree/rep_method.min_bucket_length, rep_method.num_buckets) + 1)])
		degree_counts_x = np.zeros([int(math.log(graph.max_degree/rep_method.min_bucket_length, rep_method.num_buckets) + 1)])
	else:
		degree_counts = np.zeros([(int(graph.max_degree/rep_method.min_bucket_length) + 1)])
		degree_counts_x = np.zeros ([(int(graph.max_degree/rep_method.min_bucket_length) + 1)])
	'''
	degrees = [graph.node_degrees[current_node],graph.node_degrees[current_node]*.85,graph.node_degrees[current_node]]
	for degree in degrees:
		v = int(math.log(degree / rep_method.min_bucket_length, rep_method.num_buckets));
		if(v >=0 and v< degree_counts.__len__()):
			degree_counts[ v] = 1;
	'''

	#For each node in k-hop neighbors, count its degree
	for kn in kneighbors:
		weight = 1 #unweighted graphs supported here
		degree = graph.node_degrees[kn]
		if rep_method.num_buckets is not None:
			try:
				v = math.log(degree/rep_method.min_bucket_length, rep_method.num_buckets);

				bin = int(v);
				x = math.fabs(bin + 0.5 - v);
				degree_counts_x[int(v)] += 1-x;

				if(bin + 0.5 < v and bin < degree_counts_x.__len__()):
					degree_counts_x[bin + 1] += x
				elif(bin + 0.5 > v and bin >0):
					degree_counts_x[bin - 1] += x

			except:
				print("Node %d has degree %d and will not contribute to feature distribution" % (kn, degree))
		else:
			degree_counts[int(degree/rep_method.min_bucket_length)] += weight

	#degree_counts_x = normalize(degree_counts_x[:, np.newaxis], axis=0).ravel()
	#degree_counts = normalize(degree_counts[:, np.newaxis], axis=0).ravel()

	return degree_counts_x#np.concatenate((degree_counts_x,degree_counts),axis= 0)


#Turn lists of neighbors into a degree sequence
#Input: graph, RepMethod, node's neighbors at a given layer, the node
#Output: length-D list of ints (counts of nodes of each degree), where D is max degree in graph
def get_representations_customattr(graph, rep_method, kneighbors, current_node):
	num_feature = rep_method.feature_length
	map = 1

	f = graph.__getattribute__("node_"+rep_method.featurename);
	s = -np.sort(-f[kneighbors],)
	if map > 0:
		np.append(s,np.zeros(map));
	ratios = np.zeros([num_feature]);
	ratios[0] = f[current_node];

	num_feature1= num_feature+ map;

	if (num_feature-1 < kneighbors.__len__()):
		for t in range(num_feature-1):
			#ratios[t + 1] = s[t]
			ratios[t+1] = np.mean(s[t:t+map])
	else:
		for t in range(kneighbors.__len__()):
			#ratios[t + 1] = s[t]
			ratios[t+1] = np.mean(s[t:t+map])
	

	return ratios


#Turn lists of neighbors into a degree sequence
#Input: graph, RepMethod, node's neighbors at a given layer, the node
#Output: length-D list of ints (counts of nodes of each degree), where D is max degree in graph
def get_degree_sequence_rou(graph, rep_method, kneighbors, current_node):
	minlength = 5;
	if rep_method.num_buckets is not None:
		degree_counts = [0] * int(math.log(graph.max_degree/rep_method.min_bucket_length, rep_method.num_buckets) + 1)
		degree_counts_x = [0] * int(math.log(graph.max_degree/rep_method.min_bucket_length, rep_method.num_buckets) + 1)
		rou = [0] * int(math.log(graph.max_degree/rep_method.min_bucket_length, rep_method.num_buckets) + 1)
	else:
		degree_counts = [0] * (int(graph.max_degree/rep_method.min_bucket_length) + 1)
		degree_counts_x = [0] * (int(graph.max_degree/rep_method.min_bucket_length) + 1)
		rou = [0] * (int(graph.max_degree/rep_method.min_bucket_length) + 1)


	#For each node in k-hop neighbors, count its degree
	for kn in kneighbors:
		weight = 1 #unweighted graphs supported here
		degree = graph.node_degrees[kn]
		if rep_method.num_buckets is not None:
			try:
				v = math.log(degree/rep_method.min_bucket_length, rep_method.num_buckets);
				degree_counts[int(v)] += weight
				bin = int(v);
				x = math.fabs(bin + 0.5 - v);
				degree_counts_x[int(v)] += 1-x;

				if(bin + 0.5 < v and bin < degree_counts_x.__len__()):
					degree_counts_x[bin + 1] += x
					#degree_counts[bin - 1] += weight
				elif(bin + 0.5 > v and bin >0):
					degree_counts_x[bin - 1] += x
					#degree_counts[bin - 1] += weight

			except:
				print("Node %d has degree %d and will not contribute to feature distribution" % (kn, degree))
		else:
			degree_counts[int(degree/rep_method.min_bucket_length)] += weight

	if(rep_method.num_buckets is not None):
		for t in range(degree_counts.__len__()):
			if(degree_counts[t] > 0):
				rou[t] = degree_counts_x[t]/degree_counts[t];
	else:
		rou = degree_counts;


	return rou

#Turn lists of neighbors into a degree sequence
#Input: graph, RepMethod, node's neighbors at a given layer, the node
#Output: length-D list of ints (counts of nodes of each degree), where D is max degree in graph
def get_degree_sequence_histandhiston(graph, rep_method, kneighbors, current_node):
	if rep_method.num_buckets is not None:
		degree_counts = [0] * int(math.log(graph.max_degree/rep_method.min_bucket_length, rep_method.num_buckets) + 1)
		degree_counts_x = [0] * int(math.log(graph.max_degree/rep_method.min_bucket_length, rep_method.num_buckets) + 1)
		histandhiston = [0] * int(math.log(graph.max_degree/rep_method.min_bucket_length, rep_method.num_buckets) + 1) * 2
	else:
		degree_counts = [0] * (graph.max_degree/rep_method.min_bucket_length + 1)
		degree_counts_x = [0] * (graph.max_degree/rep_method.min_bucket_length + 1)
		histandhiston = [0] * (graph.max_degree/rep_method.min_bucket_length + 1) * 2


	#For each node in k-hop neighbors, count its degree
	for kn in kneighbors:
		weight = 1 #unweighted graphs supported here
		degree = graph.node_degrees[kn]
		if rep_method.num_buckets is not None:
			try:
				v = math.log(degree/rep_method.min_bucket_length, rep_method.num_buckets);
				degree_counts[int(v)] += weight
				bin = int(v);
				x = math.fabs(bin + 0.5 - v);
				degree_counts_x[int(v)] += 1-x;

				if(bin + 0.5 < v and bin < degree_counts_x.__len__()):
					degree_counts_x[bin + 1] += x
				elif(bin + 0.5 > v and bin >0):
					degree_counts_x[bin - 1] += x

			except:
				print("Node %d has degree %d and will not contribute to feature distribution" % (kn, degree))
		else:
			degree_counts[int(degree/rep_method.min_bucket_length)] += weight
	if(rep_method.num_buckets is not None):
		histandhiston = degree_counts + degree_counts_x;
	else:
		histandhiston = degree_counts;
	return histandhiston

# Get structural features for nodes in a graph based on degree sequences of neighbors
# Input: graph, RepMethod
# Output: nxD feature matrix
def get_clusters(graph, rep_method, verbose=True):
	G_adj = graph.G_adj
	if rep_method.num_buckets is None: #1 bin for every possible degree value
		num_features = int(graph.max_degree/rep_method.min_bucket_length) + 1 #count from 0 to max degree...could change if bucketizing degree sequences

	else: #logarithmic binning with num_buckets as the base of logarithm (default: base 2)
		num_features = int(math.log(graph.max_degree/rep_method.min_bucket_length, rep_method.num_buckets)) + 1

	node_cluster = np.zeros([graph.N,num_features ])

	for n in range(graph.N):

		degrees = [graph.node_degrees[n], graph.node_degrees[n] * .85]
		for degree in degrees:
			v = int(math.log(degree / rep_method.min_bucket_length, rep_method.num_buckets));
			if (v >= 0 and v < node_cluster.__len__()):
				node_cluster[n, v] = 1;


	return node_cluster;

# Get structural features for nodes in a graph based on degree sequences of neighbors
# Input: graph, RepMethod
# Output: nxD feature matrix
def get_features_concat(graph, rep_method,feature_matrix1 = None, verbose=True):
	num_features = 0

	G_adj = graph.G_adj
	if rep_method.num_buckets is None: #1 bin for every possible degree value
		num_features = int(graph.max_degree/rep_method.min_bucket_length) + 1 #count from 0 to max degree...could change if bucketizing degree sequences
	else: #logarithmic binning with num_buckets as the base of logarithm (default: base 2)
		num_features = int(math.log(graph.max_degree/rep_method.min_bucket_length, rep_method.num_buckets)) + 1

	if(rep_method.featurename.startswith( "grarid")):
		num_features = rep_method.feature_length;
	if(feature_matrix1 is not None):
		num_features = feature_matrix1.shape[1];

	num_nodes = G_adj.shape[0]

	if rep_method.max_layer is None:
		rep_method.max_layer = graph.N #Don't need this line, just sanity prevent infinite loop
	depth = rep_method.max_layer;

	feature_matrix = np.zeros((num_nodes, num_features*depth))

	if (feature_matrix1 is not None):
		for n in range(num_nodes):
			feature_matrix[n,range(num_features)] = feature_matrix1[n,:];
	elif (rep_method.featurename == "regalid"):
		for n in range(num_nodes):
			ns = np.nonzero(graph.G_adj[n])[-1].tolist();
			feature_matrix[n,range(num_features)] = get_degree_sequence(graph, rep_method,ns , n);
	elif (rep_method.featurename.startswith("grarid")):
		for n in range(num_nodes):
			ns = np.nonzero(graph.G_adj[n])[-1].tolist();
			feature_matrix[n, range(num_features)] = get_representations_customattr(graph, rep_method, ns, n)



	for d in range(depth-1):
		for n in range(num_nodes):

			neighbors = np.nonzero(graph.G_adj[n])[-1].tolist();
			neighbors.append(n);
			#pc = 1.0/(neighbors.__len__())
			t = np.zeros(num_features);
			pc = 1.0 / (neighbors.__len__() + 1)

			for neighbor in neighbors:
				t += feature_matrix[neighbor,range(num_features*(d),num_features*(d+1))]
			feature_matrix[n, range(num_features*(d+1),num_features*(d+2))] =  pc*t


	return feature_matrix



#Get structural features for nodes in a graph based on degree sequences of neighbors
#Input: graph, RepMethod
#Output: nxD feature matrix
def get_features(graph, rep_method, verbose = True):
	before_khop = time.time()
	#Get k-hop neighbors of all nodes
	khop_neighbors_nobfs = get_khop_neighbors(graph, rep_method)

	graph.khop_neighbors = khop_neighbors_nobfs
	
	if verbose:
		print("max degree: ", graph.max_degree)
		after_khop = time.time()
		print("got k hop neighbors in time: ", after_khop - before_khop)

	G_adj = graph.G_adj
	num_nodes = G_adj.shape[0]
	if rep_method.num_buckets is None: #1 bin for every possible degree value
		num_features = int(graph.max_degree/rep_method.min_bucket_length) + 1 #count from 0 to max degree...could change if bucketizing degree sequences
	else: #logarithmic binning with num_buckets as the base of logarithm (default: base 2)
		num_features = int(math.log(graph.max_degree/rep_method.min_bucket_length, rep_method.num_buckets)) + 1
	feature_matrix = np.zeros([num_nodes, num_features])

	before_degseqs = time.time()
	for n in range(num_nodes):
		for layer in graph.khop_neighbors[n].keys(): #construct feature matrix one layer at a time
			if len(graph.khop_neighbors[n][layer]) > 0:
				#degree sequence of node n at layer "layer"
				deg_seq = get_degree_sequence(graph, rep_method, graph.khop_neighbors[n][layer], n)
				#add degree info from this degree sequence, weighted depending on layer and discount factor alpha
				feature_matrix[n] += [(rep_method.alpha**layer) * x for x in deg_seq]
	after_degseqs = time.time() 

	if verbose:
		print ("got degree sequences in time: ", after_degseqs - before_degseqs)

	return feature_matrix

def compute_shortestpath(graph,sources):
	g = graph.tempgraph

	for source in sources:
		g.add_edge(source[0],source[1],weight=0);

	sh = np.multiply(graph.N/2 , np.ones((graph.N,sources.shape[0])))
	si = 0
	for source in sources:
		bfs = [0]*graph.N;
		ns = []
		for i in range(2):
			ns.append(source[i])
			sh[source[i],si] = 0
			bfs[source[i]] = 1

		while ns.__len__() > 0:
			n = ns.pop(0)
			for nn in g.neighbors(n):
				if(bfs[nn] == 0):
					ns.append(nn);
					bfs[nn] = 1
					sh[nn,si] = sh[n,si]+g[n][nn]['weight']
		si += 1
	'''
	index = 0
	for source in sources:
		sh[source[0],:] = np.multiply(graph.N/2 , np.ones(sources.shape[0]))
		sh[source[1], :] = np.multiply(graph.N/2 , np.ones(sources.shape[0]))
		sh[source[1], index] = 0
		sh[source[0], index] = 0
		index = index + 1
	'''
	return  sh;


#Input: two vectors of the same length
#Optional: tuple of (same length) vectors of node attributes for corresponding nodes
#Output: number between 0 and 1 representing their similarity
def compute_similarity(graph, rep_method, vec1, vec2, node_attributes = None, node_indices = None):
	dist = rep_method.gammastruc * np.linalg.norm(vec1 - vec2) #compare distances between structural identities
	if graph.node_attributes is not None:
		#distance is number of disagreeing attributes 
		attr_dist = np.sum(graph.node_attributes[node_indices[0]] != graph.node_attributes[node_indices[1]])
		dist += rep_method.gammaattr * attr_dist
	return np.exp(-dist) #convert distances (weighted by coefficients on structure and attributes) to similarities

#Sample landmark nodes (to compute all pairwise similarities to in Nystrom approx)
#Input: graph (just need graph size here), RepMethod (just need dimensionality here)
#Output: np array of node IDs
def get_sample_nodes(graph, rep_method, verbose = True):
	#Sample uniformly at random
	sample = np.random.permutation(np.arange(graph.N))[:rep_method.p]
	return sample

#Get dimensionality of learned representations
#Related to rank of similarity matrix approximations
#Input: graph, RepMethod
#Output: dimensionality of representations to learn (tied into rank of similarity matrix approximation)
def get_feature_dimensionality(graph, rep_method, verbose = True):
	p = int(rep_method.k*math.log(graph.N/rep_method.min_bucket_length, rep_method.num_buckets)) #k*log(n) -- user can set k, default 10
	if verbose:
		print ("feature dimensionality is "+ str(min(p, graph.N)))
	rep_method.p = min(p,graph.N)  #don't return larger dimensionality than # of nodes
	return rep_method.p

#xNetMF pipeline
def get_representations(graph, rep_method,feature_matrix, verbose = True):


	#Efficient similarity-based representation
	#Get landmark nodes
	if rep_method.p is None:
		rep_method.p = get_feature_dimensionality(graph, rep_method, verbose = verbose) #k*log(n), where k = 10
	elif rep_method.p > graph.N: 
		print ("Warning: dimensionality greater than number of nodes. Reducing to n")
		rep_method.p = graph.N


	landmarks = get_sample_nodes(graph, rep_method, verbose=verbose)
	#Explicitly compute similarities of all nodes to these landmarks
	before_computesim = time.time()
	C = np.zeros((graph.N,rep_method.p))
	for node_index in range(graph.N): #for each of N nodes
		for landmark_index in range(rep_method.p): #for each of p landmarks
			#select the p-th landmark
			C[node_index,landmark_index] = compute_similarity(graph, 
															rep_method, 
															feature_matrix[node_index], 
															feature_matrix[landmarks[landmark_index]], 
															graph.node_attributes, 
															(node_index, landmarks[landmark_index]))



	#Compute Nystrom-based node embeddings
	W_pinv = np.linalg.pinv(C[landmarks])
	U,X,V = np.linalg.svd(W_pinv)
	Wfac = np.dot(U, np.diag(np.sqrt(X)))
	reprsn = np.dot(C, Wfac)



	#Post-processing step to normalize embeddings (true by default, for use with REGAL)
	if rep_method.normalize:
		reprsn = reprsn / np.linalg.norm(reprsn, axis = 1).reshape((reprsn.shape[0],1))
	return reprsn


def get_representations_grar(graph, rep_method,feature_matrix, verbose=True):



	# Efficient similarity-based representation
	# Get landmark nodes

	if rep_method.p is None:
		rep_method.p = get_feature_dimensionality(graph, rep_method, verbose=verbose)  # k*log(n), where k = 10
	elif rep_method.p > graph.N:
		print("Warning: dimensionality greater than number of nodes. Reducing to n")
		rep_method.p = graph.N

	if rep_method.add_cluster_feature == True:
		cluster_attrs = get_clusters(graph, rep_method)
		cluster_attrs = normalize(cluster_attrs, axis=1, norm='l2')

		feature_matrix = normalize(feature_matrix, axis=1, norm='l2')
		feature_matrix = np.concatenate((cluster_attrs, feature_matrix), axis=1)

	if(rep_method.normalize):
		feature_matrix = normalize(feature_matrix, axis=1, norm='l2')
	#add extra attribute


	if graph.node_attributes is not None :
		#graph.node_attributes = get_features_concat(graph, rep_method, feature_matrix1=graph.node_attributes)
		feature_matrix = np.concatenate((graph.node_attributes,feature_matrix),axis=1)

	if(rep_method.framework == "none"):
		return feature_matrix

	emb1 ,emb2 = alignments.get_embeddings(feature_matrix);

	alignment_matrix = alignments.get_embedding_similarities(emb1, emb2, num_top=1)
	#maxel = math.ceil(rep_method.p / graph.component_number)
	maxel = 1
	numel = np.zeros([graph.component_number]);
	bestmatch = []
	bestmatchindex = []
	iscorrect = []


	for x in range(emb1.shape[0]):
		row, possible_alignments, possible_values = sp.find(alignment_matrix[x])
		bestmatch.append(possible_values[0]);
		bestmatchindex.append([x,possible_alignments[0] + emb1.shape[0]]);
		iscorrect.append(np.sum(graph.true_alignments[x] == possible_alignments ) > 0)
	bins = [_i for _i in range(50,100)];
	z=plt.hist(np.multiply(bestmatch,100),bins = bins);
	maxind = np.argmax(z[0])
	m1 = np.sum(z[0][maxind:]) / np.sum(z[0][0:maxind])
	cnf.par1 = z

	numcandid = np.sum(np.array(bestmatch)>=0.98);
	if(numcandid > rep_method.p ):

		t = np.argsort(-np.array(bestmatch))[:numcandid]
		random.shuffle(t)
		sortedlist = t[:rep_method.p];

		landmarks_tuple = np.array(bestmatchindex)[sortedlist[:rep_method.p]]
		landmarks_tuple_ =  np.unique(np.array([graph.component_index[x[0]] for x in landmarks_tuple]+[graph.component_index[x[1]] for x in landmarks_tuple]));
		sh = compute_shortestpath(graph,np.array(landmarks_tuple))
		sh = np.float_power(np.add(sh,1),-1)
		sh = normalize(sh, axis=1, norm='l2')
	else:
		sh = feature_matrix
	'''
	color_map = []
	for node in graph.graph:
		if sortebestmatch1.__contains__(node) or sortebestmatch2.__contains__(node):
			color_map.append('green')
		else:
			color_map.append('blue')

	nx.draw(graph.graph, node_color=color_map,width=1, with_labels=False)
	plt.show()
	
	flen = feature_matrix[0,:].__len__();
	for t in range(graph.N):
		if(landmarks_tuple_.__contains__(graph.component_index[t])==False):
			feature_matrix[t,:]= np.zeros([flen])
	'''
	#sh = np.concatenate((sh,feature_matrix),axis=1);
	return sh

def get_representations_all(graph, rep_method, verbose=True):
	# Node identity extraction
	before_computefeature = time.time()

	if (rep_method.featuremethod == "concat"):
		feature_matrix = get_features_concat(graph, rep_method, verbose=verbose)
	elif (rep_method.featuremethod == "normal"):
		feature_matrix = get_features(graph, rep_method, verbose=verbose)
	after_computefeature = time.time()

	if verbose:
		print("computed feature in time: ",after_computefeature-before_computefeature)

	index = 0
	output=[]

	for framework in rep_method.frameworks:
		before_computerep = time.time()
		rep_method.framework = framework;
		if(framework == 'regal'):
			_framework_feature = get_representations(graph,rep_method,feature_matrix, verbose=verbose)
		elif(framework == 'grar'):
			_framework_feature = get_representations_grar(graph,rep_method,feature_matrix, verbose=verbose)
		elif(framework == 'none'):
			_framework_feature = feature_matrix;
			if rep_method.add_cluster_feature == True:
				cluster_attrs = get_clusters(graph, rep_method)
				cluster_attrs = normalize(cluster_attrs, axis=1, norm='l2')

				_framework_feature = normalize(_framework_feature, axis=1, norm='l2')
				_framework_feature = np.concatenate((cluster_attrs, _framework_feature), axis=1)

			if graph.node_attributes is not None:
				# graph.node_attributes = get_features_concat(graph, rep_method, feature_matrix1=graph.node_attributes)
				_framework_feature = np.concatenate((graph.node_attributes, _framework_feature), axis=1)


		after_computerep = time.time()
		if(framework == 'none'):
			after_computerep = before_computerep;
		print("computed framework feature in time: ", after_computerep-before_computerep)

		output.append([ _framework_feature,(after_computefeature-before_computefeature),(after_computerep - before_computerep)])
		index = index + 1

	return output



if __name__ == "__main__":
	if len(sys.argv) < 2:
		#####PUT IN YOUR GRAPH AS AN EDGELIST HERE (or pass as cmd line argument)#####  
		#(see networkx read_edgelist() method...if networkx can read your file as an edgelist you're good!)
		graph_file = "data/arenas_combined_edges.txt"
	else:
		graph_file = sys.argv[1]
	nx_graph = nx.read_edgelist(graph_file, nodetype = int, comments="%")
	adj_matrix = nx.adjacency_matrix(nx_graph).todense()
	
	graph = Graph(adj_matrix)
	rep_method = RepMethod(max_layer = 2) #Learn representations with xNetMF.  Can adjust parameters (e.g. as in REGAL)
	representations = get_representations(graph, rep_method)
	print(representations.shape)





