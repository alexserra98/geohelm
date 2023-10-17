import numpy as np
DTYPE = np.float32
cpdef float neig_overlap(long[:,:] X,long[:,:] Y):
  """
  Computes the neighborhood overlap between two representations.
  Parameters
  ----------
  X : 2D array of ints
      nearest neighbor index matrix of the first representation
  Y : 2D array of ints
      nearest neighbor index matrix of the second representation
  
  Returns
  -------
  overlap : float
      neighborhood overlap between the two representations
  """
  assert X.shape[0] == Y.shape[0]
  cdef Py_ssize_t ndata = X.shape[0]
  cdef Py_ssize_t k = X.shape[1]
  cdef int i
  overlaps = np.empty(ndata, dtype=DTYPE)
  cdef float[:] overlaps_memview = overlaps
  cdef long[:,:] X_memview = X 
  cdef long[:,:] Y_memview = Y 
  
  for i in range(ndata):
    overlaps_memview[i] = np.intersect1d(X_memview[i],Y_memview[i]).shape[0]/k
  return np.mean(overlaps)

cpdef float[:,:] _instances_overlap(long[:,:,:] nn1,long[:,:,:] nn2):
  #print(nn1.shape,nn2.shape)
  assert nn1.shape[0] == nn2.shape[0], "The two nearest neighbour matrix must have the same shape" 
  cdef Py_ssize_t layers_len = nn1.shape[0]
  overlaps = np.empty([layers_len, layers_len], dtype=DTYPE)
  cdef float[:,:] overlaps_memview = overlaps
  cdef long[:,:,:] nn1_memview = nn1
  cdef long[:,:,:] nn2_memview = nn2
  cdef int i,j
  for i in range(layers_len):
    for j in range(layers_len):
      # WARNING : the overlap is computed with K=K THE OTHER OCC IS IN RUNGEOMETRY
      overlaps_memview[i][j] = neig_overlap(nn1_memview[i], nn2_memview[j])
  return overlaps