## a GPU version of the fisher_metric.py
# use torch 
# 
# 
import numpy as np
import math
import cupy as cp
from tqdm import tqdm
# def gamma(x, y, t, axis_x=0, axis_y=0):
# 	N = len(t)
# 	x_shape = x.shape[1:]
# 	rank = len(x_shape)
# 	t_rep = cp.reshape(t, [N] + [1]*rank)

# 	for i, dim in enumerate(x_shape):
# 		t_rep = t_rep.repeat(dim, i+1)

# 	t_rep1 = cp.reshape(t, [1, N] + [1]*rank)
# 	t_rep1 = t_rep1.repeat(len(y), 0)

# 	for i, dim in enumerate(x_shape):
# 		t_rep1 = t_rep1.repeat(dim, i+2)

# 	# vectors 10 x 3 x 32 x 32
# 	x_rep = cp.repeat(x, N, axis_x)
# 	# vectors 5 x 10 x 3 x 32 x 32
# 	y_rep = cp.repeat(y, N, axis_y)
# 	# vectors 5 x 10 x 3 x 32 x 32
# 	return x_rep*t_rep + (1-t_rep1)*y_rep
def gamma(x, y, t, axis_x=0, axis_y=0):
    N = len(t)
    x_shape = x.shape[1:]
    rank = len(x_shape)

    # create tensor of shape (N, x.shape[1], x.shape[2], ..., x.shape[-1])
    t_rep = t.reshape((N,) + (1,) * rank)

    # broadcast t along each dimension of x except for the first axis
    x_rep = x * t_rep

    # create tensor of shape (N, y.shape[1], y.shape[2], ..., y.shape[-1])
    t_rep1 = t.reshape((1, N) + (1,) * rank)

    # broadcast t along each dimension of y except for the first two axes
    y_rep = y * (1 - t_rep1)

    return x_rep + y_rep

def p_ni_row(x, y, n, i):
	return gamma(x, y, (i/n), axis_x=0, axis_y=1)

def kl_divergence(p, q, axis):
	# add epsilon for numeric stability
	p += 1e-10
	q += 1e-10
	return cp.sum(cp.where(p != 0, p * cp.log(p / q), 0), axis=axis)
    
def d_js(p, q, axis=1):
	m = (p + q)/2.   
	kl1 = kl_divergence(p, m, axis=axis)
	kl2 = kl_divergence(q, m, axis=axis)
	return 0.5 * (kl1 + kl2)

def euclidian_distance(x, y, axis=(1, 2, 3)):
	'''This corresponds to d_s in the paper'''
	#return cp.sqrt(cp.sum((x - y)**2, axis=axis))
	diff = (x - y).reshape(len(y),-1)
	return cp.linalg.norm(diff, axis=-1)

def predict_many(model, x, n_classes, batch_size):
	# x -> (row_len, interpol, data_shape)
	orig_shape = cp.shape(x)

	# x -> (row_len * interpol, data_shape)
	x_reshape = cp.vstack(x)
	
	n_inputs = len(x_reshape)
	# p -> (40, 10, 10)
	preds = cp.zeros([len(x_reshape), n_classes])
	
	n_batches = max(math.ceil(n_inputs/batch_size), 1)

	for b in range(n_batches):
		r1, r2 = b*batch_size, (b+1)*batch_size
		inputs = x_reshape[r1:r2]
		pred = model(inputs)
		# pred = cp.asarray(pred)
		preds[r1:r2] = pred
	
	np_preds = cp.vsplit(preds, orig_shape[0])
	return cp.array(np_preds)

def distance_row(model, x, y, n, batch_size, n_classes):
	y = y[:,cp.newaxis]

	steps = cp.arange(1, n+2)
	sprev = steps-1 #cp.where(steps-1 < 0, 0, steps-1)
    
	p_prev = p_ni_row(x, y, n+1, sprev)
	p_i = p_ni_row(x, y, n+1, steps)
	
	djs = d_js(predict_many(model, p_prev, n_classes, batch_size),
			   predict_many(model, p_i, n_classes, batch_size), axis=2)
	
	# distance measure based on classification
	discriminative = cp.sqrt(cp.maximum(djs, 0.))
	# euclidian distance measure based on structural differences
	axes = tuple(range(2, len(x.shape)+1))

	euclidian = euclidian_distance(x, y, axis=axes)
	
	return discriminative.sum(axis=1), euclidian

def calculate_fisher(model, from_samples, to_samples, n, batch_size, n_classes, verbose=True):
	

	n_xs = len(from_samples)
	n_ys = len(to_samples)

	from_samples = cp.asarray(from_samples)
	to_samples = cp.asarray(to_samples)

	# arrays to store distance
	#  1. discriminative distance of classification
	#  2. euclidian (structural) distance in data
	discr_distances = cp.zeros([n_xs, n_ys])
	eucl_distances = cp.zeros([n_xs, n_ys])

	for i in tqdm(range(n_xs)):

		x = from_samples[i]
		x = x[cp.newaxis]
		ys = to_samples[i+1:]
	
		disc_row = cp.zeros(n_ys)
		eucl_row = cp.zeros(n_ys)
		
		if len(ys) != 0:
			discr, euclidian = distance_row(model, x, ys, n, batch_size, n_classes)
			disc_row[i+1:] = discr
			eucl_row[i+1:] = euclidian

		discr_distances[i] = disc_row
		eucl_distances[i] = eucl_row

		if (i+1) % (n_xs//5) == 0:
			if verbose:
				print('Distance calculation %.2f %%' % (((i+1)/n_xs)*100), end='\t')
				print('samples: ', str(i+1), '/', str(n_xs))

	return discr_distances.get(), eucl_distances.get()

def calculate_fisher_new(model, from_samples, to_samples, n, batch_size, n_classes, verbose=True):

	"""
	Do not concat the new samples to the old ones	
	"""
    # move to gpu
	from_samples = cp.asarray(from_samples)
	to_samples = cp.asarray(to_samples)

	n_xs = len(from_samples)
	n_ys = len(to_samples)

	# arrays to store distance
	#  1. discriminative distance of classification
	#  2. euclidian (structural) distance in data
	discr_distances = cp.zeros([n_xs, n_ys])
	eucl_distances = cp.zeros([n_xs, n_ys])

	# for i in range(n_xs):
	# use tyqdm to show progress
	for i in tqdm(range(n_xs)):
		
		x = from_samples[i]
		x = x[cp.newaxis]
		ys = to_samples
	
		# if len(ys) != 0:
		discr, euclidian = distance_row(model, x, ys, n, batch_size, n_classes)
		# disc_row[i] = discr
		# eucl_row[i] = euclidian

		discr_distances[i] = discr
		eucl_distances[i] = euclidian

		if (i+1) % (n_xs//5) == 0:
			if verbose:
				print('Distance calculation for transforming %.2f %%' % (((i+1)/n_xs)*100), end='\t')
				print('samples: ', str(i+1), '/', str(n_xs))

	return cp.asnumpy(discr_distances), cp.asnumpy(eucl_distances)