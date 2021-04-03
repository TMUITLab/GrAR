import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import time
import os
import config as cnf
import gen_combined
import sys
try: import cPickle as pickle
except ImportError:
	import pickle
from scipy.sparse import csr_matrix
pd.set_option('display.max_columns', 12)

import xnetmf
from config import *
from alignments import *

def parse_args():
	parser = argparse.ArgumentParser(description="Run REGAL.")
	parser.add_argument('--datasetname', nargs='?', default='noname',
						help="datasetname")

	parser.add_argument('--input', nargs='?', default='datasets/arenas/combined01.txt', help="Edgelist of combined input graph")

	parser.add_argument('--output', nargs='?', default='datasets/arenas/graph01.txt',
	                    help='Embeddings path')

	parser.add_argument('--attributes', nargs='?', default='data/attributes/attr1-2vals-prob0.000000',
	                    help='File with saved numpy matrix of node attributes, or int of number of attributes to synthetically generate.  Default is 5 synthetic.')

	parser.add_argument('--attrvals', type=int, default=10,
	                    help='Number of attribute values. Only used if synthetic attributes are generated')

	parser.add_argument('--add_cluster_feature', type=bool, default=False,
	                    help='add_cluster_feature')

	parser.add_argument('--feature_length', type=int, default=10,
	                    help='Number of feature_length. Default is 128.')

	parser.add_argument('--k', type=int, default=10,
	                    help='Controls of landmarks to sample. Default is 10.')

	parser.add_argument('--untillayer', type=int, default=4,
                    	help='Calculation until the layer for xNetMF.')
	parser.add_argument('--alpha', type=float, default = 0.1, help = "Discount factor for further layers")
	parser.add_argument('--gammastruc', type=float, default = 1, help = "Weight on structural similarity")
	parser.add_argument('--gammaattr', type=float, default = 1, help = "Weight on attribute similarity")
	parser.add_argument('--numtop', type=int, default=10,help="Number of top similarities to compute with kd-tree.  If 0, computes all pairwise similarities.")
	parser.add_argument('--buckets', default=2, type=float, help="base of log for degree (node feature) binning")
	parser.add_argument('--min_bucket_length', default=1, type=float, help="base of log for degree (node feature) binning")
	parser.add_argument('--featurename', default='histon',  help="feature that used for alignment : histogram , histon , rou")
	parser.add_argument('--featuremethod', default='normal',  help="methods used to compute feature : normal , fastattch , fastmerge ")
	parser.add_argument('--framework', default='regal',  help="framework to use : regal , tsol")

	return parser.parse_args()

def main(args):
	dataset_name = args.input.split("/")
	if len(dataset_name) == 1:
		dataset_name = dataset_name[-1].split(".")[0]
	else:
		dataset_name = dataset_name[-2]

	#Get true alignments
	true_alignments_fname = args.input.replace('combined','truth');
	print ("true alignments file: " + true_alignments_fname)
	true_alignments = None
	if os.path.exists(true_alignments_fname):
		with open(true_alignments_fname, "rb") as true_alignments_file:
			true_alignments = pickle.load(true_alignments_file ,fix_imports=True, encoding='latin1')
		args.true_alignments = true_alignments

	#Load in attributes if desired (assumes they are numpy array)
	if args.attributes is not None:
		args.attributes = np.load(args.attributes) #load vector of attributes in from file
		print (args.attributes.shape)

	#Learn embeddings and save to output
	print ("learning representations...")
	before_rep = time.time()
	feature_computation_time,framewok_computation_time = learn_representations(args)
	after_rep = time.time()
	print("Learned representations in %f seconds" % (after_rep - before_rep))
	index = 0
	finaltime = [];
	output_d = []
	for frm in args.frameworks:
		ot = str(frm) +'-'+ args.featuremethod + '-' + args.featurename

		embed = np.load(args.output+ot,allow_pickle=True)
		emb1, emb2 = get_embeddings(embed)
		before_align = time.time()
		if args.numtop == 0:
			args.numtop = None
		if(args.numtop > emb1.shape[0]):
			args.numtop = emb1.shape[0];
		alignment_matrix = get_embedding_similarities(emb1, emb2, num_top = args.numtop)

		#Report scoring and timing
		after_align = time.time()
		total_time = after_align - before_align

		d = np.zeros(5);
		if true_alignments is not None:
			topk_scores = [1,5,10,20,50]
			ind = 0
			for k in topk_scores:
				score, correct_nodes = score_alignment_matrix(alignment_matrix, topk = k, true_alignments = true_alignments)
				print("score top%d: %f" % (k, score))
				d[ind] = score;
				ind = ind + 1
		output_d.append(d)
		finaltime.append([feature_computation_time[index],framewok_computation_time[index]])
		print(ot, "align time ", ":", total_time)
		print(ot,"feature time " , ":", feature_computation_time[index])
		print(ot, "framework time ",":", framewok_computation_time[index])
		index = index + 1
	return  output_d,finaltime
#Should take in a file with the input graph as edgelist (args.input)
#Should save representations to args.output
def learn_representations(args):
	nx_graph = nx.read_edgelist(args.input, nodetype = int, comments="%")#,create_using=nx.DiGraph())#read graph)

	#xnetmf.plotgraph(nx_graph);
	print ("read in graph")
	adj = nx.adjacency_matrix(nx_graph,range(nx_graph.nodes.__len__()))#.todense()
	print ("got adj matrix")

	graph = Graph(adj, node_attributes = args.attributes,graph=nx_graph,true_alignments=args.true_alignments)
	max_layer = args.untillayer
	if args.untillayer == 0:
		max_layer = None
	alpha = args.alpha
	num_buckets = args.buckets #BASE OF LOG FOR LOG SCALE
	if num_buckets == 1:
		num_buckets = None
	rep_method = RepMethod(max_layer = max_layer,
							alpha = alpha,
							k = args.k,
							add_cluster_feature = args.add_cluster_feature,
							feature_length= args.feature_length,
							min_bucket_length= args.min_bucket_length,
							featuremethod = args.featuremethod,
							featurename = args.featurename,
							frameworks = args.frameworks,
							num_buckets = num_buckets,
							normalize = False,
							gammastruc = args.gammastruc,
							gammaattr = args.gammaattr)
	if max_layer is None:
		max_layer = 1000
	print("Learning representations with max layer %d ,alpha = %f ,featuremethod = %s , featurename = %s and dataset = %s" % (max_layer, alpha,rep_method.featurename,rep_method.featuremethod,args.datasetname))


	outputs = xnetmf.get_representations_all(graph, rep_method)
	feature_time = []
	framework_time = []
	index = 0
	for frm in args.frameworks:
		pickle.dump(outputs[index][0], open(args.output+ frm+'-'+
										  rep_method.featuremethod + "-" + rep_method.featurename, "wb"))
		feature_time.append(outputs[index][1])
		framework_time.append(outputs[index][2])
		index = index + 1


	return feature_time,framework_time
	#except:
	#	print('error')

if __name__ == "__main__":

	args = parse_args()

	data = {
		"add_cluster_feature": [],
		"run": [],
		"datasetname": [],
		"framework": [],
		"feature": [],
		"ps": [],
		"pa": [],
		"method": [],
		"num_attr":[],
		"framework_runtime": [],
		"feature_runtime": [],
		"top1": [],
		"top5": [],
		"top10": [],
		"top20": [],
		"top50": [],
		"k":[],

		#"par1":[],
		#"par2":[],
		#"par3":[],
		#"par4": [],
		#"par5": []
	}

	for datasetname in ['arenas','ppi','dblp','arxiv'] :
		for run in range(0,1):
			percents = [0.01]
			apercents = [None]
			dif_ks = [10]
			numattrs = [None]
			methods = ['concat','normal']
			features =['grarid','regalid']
			frameworks = ['grar','regal','none']
			pind = 0;

			for ps in percents:
				#gen_combined.gen_combined_edges(dataset_name=datasetname, percent=ps)
				#continue
				pa = apercents[0]
				for dif_k in dif_ks:
					args.k = dif_k
					for numattr in numattrs:
						truth_file = 'datasets/' + datasetname + "/truth00"+ "{:02d}".format(int(ps*100)) + '.txt'
						if(pa is not None):
							gen_combined.gen_attribute(dataset_name=datasetname, pa=pa,num_attr=numattr, truth_file=truth_file )
						for m in methods:
							custom_features = features
							if(m == "normal"):
								custom_features = ['hist'];
							for f in custom_features:

								for ac in range(1,2):
									if(ac==1):
										args.add_cluster_feature = True;
									else:
										args.add_cluster_feature = False;

									args.datasetname = datasetname  + "{:02d}".format(int(ps*100));
									args.featurename = f
									args.featuremethod = m
									args.frameworks = frameworks
									args.input = 'datasets/' + datasetname + "/combined00"+ "{:02d}".format(int(ps*100)) + '.txt'
									args.output = 'datasets/' + datasetname + '/'
									if(pa is None):
										args.attributes = None;
									else:
										args.attributes = 'datasets/' + datasetname + "/attr00"+ "{:02d}".format(int(ps*100)) + '.txt' + "{:02d}".format(int(pa*100))
									_r, _t = main(args)

									while(True):
										try:

											break
										except :
											gen_combined.gen_combined_edges(dataset_name=datasetname, percent=ps)

									index = 0
									for frm in frameworks:
										data["num_attr"].append(numattr )
										data["add_cluster_feature"].append(ac )
										data["run"].append(run);
										data["datasetname"].append(datasetname )
										data["framework"].append(frm)
										data["k"].append(dif_k)
										data["feature"].append(f)
										data["ps"].append(ps)
										data["pa"].append(pa)
										data["method"].append(m)
										data["framework_runtime"].append(_t[index][1])
										data["feature_runtime"].append(_t[index][0])
										data["top1"].append(_r[index][0])
										data["top5"].append(_r[index][1])
										data["top10"].append(_r[index][2])
										data["top20"].append(_r[index][3])
										data["top50"].append(_r[index][4])


										index = index + 1
										plt.clf();

									df = pd.DataFrame(data);
									df.to_csv('outputattr1' + str(run) + '.csv');

	df = pd.DataFrame(data);
	print(df[["framework", "feature", "method", "ps", "pa", "top1", "top5", "top10",
			  "top20", "top50","k"]])

