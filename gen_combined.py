import random
import pickle
import networkx as nx
import scipy.sparse
from scipy.sparse import coo_matrix
import networkx
import numpy

def gen_combined_edges(dataset_name='arenas',percent = 0.01,spercent = 0):
	outputdir = 'datasets/'+dataset_name + '/' ;
	edges = numpy.loadtxt(outputdir + 'graph.txt', dtype=int)
	edges_len = edges.__len__()
	vs = numpy.reshape(edges,(2*edges_len,1))
	nodes_dic = dict()
	for v in vs:
		nodes_dic[v[0]] = 1
	nodes = [x for x in nodes_dic.keys()]
	length = nodes.__len__()  # number of nodes

	I = numpy.zeros([numpy.max(nodes)+1],dtype=int)
	P = numpy.zeros([numpy.max(nodes)+1],dtype=int)
	degrees = [0]*length;
	P1 = dict()
	Plist = list(range(0,length))
	random.shuffle(Plist);
	for i in range(nodes.__len__()):
		I[nodes[i]]=i
		P[i] = Plist[i]
		P1[i] = Plist[i]
	for v in vs:
		degrees[I[v[0]]] += 1

	with open(outputdir + 'truth'+ "{:02d}".format(int(spercent*100)) + "{:02d}".format(int(percent*100))+'.txt', 'wb') as f:  # this file contain the true alignment inforamation
		pickle.dump(P1, f)

	for i in range(nodes.__len__()):
		P[i] = Plist[i] + nodes.__len__()

	n_edgelist = []
	tedges = []
	for edge in edges:
		tedges.append([I[edge[0]],I[edge[1]]]);

	for k in range(2):
		deleted_edges = []
		samps = []
		tdegrees = degrees.copy()
		numsamples = 0
		if(k==1):
			if (percent > 0):
				numsamples =  int(percent * edges_len)
				#samps = random.sample(range(edges_len),)
		else:
			if(spercent > 0):
				numsamples = int(spercent * edges_len)
				#samps = random.sample(range(edges_len), int(spercent * edges_len))

		while numsamples > 0:
			samp = int(random.sample(range(edges_len),1)[0])
			if(deleted_edges.__contains__(samp) == False and tdegrees[tedges[samp][1]] > 1 and tdegrees[tedges[samp][0]] > 1 ):
				tdegrees[tedges[samp][1]] -= 1
				tdegrees[tedges[samp][0]] -= 1
				deleted_edges.append(samp)
				numsamples -= 1

		aprimedges = list(range(edges_len))
		for x in deleted_edges:
			aprimedges.remove(x);
		if k== 1:
			for x in aprimedges:
				n_edgelist.append([P[tedges[x][0]], P[tedges[x][1]]]);
		else:
			for x in aprimedges:
				n_edgelist.append(tedges[x]);

	for i in range(len(n_edgelist)):
		n_edgelist[i].append({'weight': 1.0})

	data = numpy.array(n_edgelist)

	# this ouput file still has some character like '[' ']' ','... that needed to be deleted by manual operation
	numpy.savetxt(outputdir + "combined" + "{:02d}".format(int(spercent * 100)) + "{:02d}".format(int(percent * 100))+".txt", data, delimiter=" ", fmt='%s')
	#end
def gen_combined_edges_random(n,p , dataset_name='random',percent = 0.03,spercent = 0):

	outputdir = 'datasets/'+random + '/' + str(n) + '-'+str(p)  ;
	g = networkx.erdos_renyi_graph(n,p);
	edges = []
	for x, y in g.edges:
		edges.append((x,y));

	edges_len = edges.__len__()
	vs = numpy.reshape(edges,(2*edges_len,1))
	nodes_dic = dict()
	for v in vs:
		nodes_dic[v[0]] = 1
	nodes = [x for x in nodes_dic.keys()]
	length = nodes.__len__()  # number of nodes

	I = numpy.zeros([numpy.max(nodes) + 1], dtype=int)
	P = numpy.zeros([numpy.max(nodes) + 1], dtype=int)
	degrees = [0] * length;
	P1 = dict()
	Plist = list(range(0, length))
	random.shuffle(Plist);
	for i in range(nodes.__len__()):
		I[nodes[i]] = i
		P[i] = Plist[i]
		P1[i] = Plist[i]
	for v in vs:
		degrees[I[v[0]]] += 1

	with open(outputdir + 'truth' + "{:02d}".format(int(spercent * 100)) + "{:02d}".format(int(percent * 100)) + '.txt',
			  'wb') as f:  # this file contain the true alignment inforamation
		pickle.dump(P1, f)

	for i in range(nodes.__len__()):
		P[i] = Plist[i] + nodes.__len__()

	n_edgelist = []
	tedges = []
	for edge in edges:
		tedges.append([I[edge[0]], I[edge[1]]]);

	for k in range(2):
		deleted_edges = []
		samps = []
		tdegrees = degrees.copy()
		if (k == 1):
			if (percent > 0):
				numsamples = int(percent * edges_len)
				# samps = random.sample(range(edges_len),)
		else:
			if (spercent > 0):
				numsamples = int(spercent * edges_len)
				# samps = random.sample(range(edges_len), int(spercent * edges_len))

		while numsamples > 0:
			samp = int(random.sample(range(edges_len),1)[0])
			if (deleted_edges.__contains__(samp) == False and tdegrees[tedges[samp][1]] > 1 and tdegrees[
				tedges[samp][0]] > 1):
				tdegrees[tedges[samp][1]] -= 1
				tdegrees[tedges[samp][0]] -= 1
				deleted_edges.append(samp)
				numsamples -= 1


		aprimedges = list(range(edges_len))
		for x in deleted_edges:
			aprimedges.remove(x);
		if k == 1:
			for x in aprimedges:
				n_edgelist.append([P[tedges[x][0]], P[tedges[x][1]]]);
		else:
			for x in aprimedges:
				n_edgelist.append(tedges[x]);


	for i in range(len(n_edgelist)):
		n_edgelist[i].append({'weight': 1.0})

	data = numpy.array(n_edgelist)

	# this ouput file still has some character like '[' ']' ','... that needed to be deleted by manual operation
	numpy.savetxt(outputdir + "combined" + "{:02d}".format(int(spercent * 100)) + "{:02d}".format(int(percent*100))+".txt", data, delimiter=" ", fmt='%s')
	#
def gen_attribute(dataset_name='random',truth_file='',pa= 0.01,num_attr = 1,attr_val = 2):
	outputdir = 'datasets/'+dataset_name + '/' ;
	with open(truth_file, 'rb') as f:  # this file contain the true alignment inforamation
		P = pickle.load( f,fix_imports=True, encoding='latin1')
	num_element = P.keys().__len__();

	attrs = numpy.zeros([num_element*2,num_attr]);
	for attr in range(num_attr):
		attrs[:num_element,attr] = numpy.array([random.randrange(0, attr_val, 1) for _ in range(num_element)])

		samps = random.sample(range(num_element), int(pa * num_element))

		for i in range(num_element):
			attrs[num_element + P[i],attr] = attrs[i,attr]
		for samp in samps:
			attrs[num_element + P[samp],attr] = (attrs[samp,attr] + 1) % attr_val

	data = numpy.array(attrs)
	with open(truth_file.replace('truth','attr')+"{:02d}".format(int(pa * 100)), 'wb') as f:
		numpy.save( f, attrs)


if __name__ == "__main__":
	datasets = ['arxiv' ]#,'arxiv','facebook']
	percents = [0.04]#,0.02,0.03,0.04,0.05,0.1,0.2,0.3,0.4]

	for d in datasets:
		for pc in percents:
			#gen_combined_edges(dataset_name='arenas',percent=0.01)
			print('database:' + d + ' percent' +str(pc) + ' source percent : 00' )
			gen_combined_edges(dataset_name=d,percent=pc)
			pa = 0.01
			truth_file = 'datasets/' + d + '/' + "truth00" +"{:02d}".format(int(pc * 100))+".txt";
			gen_attribute(dataset_name=d,pa = pa,truth_file=truth_file)

