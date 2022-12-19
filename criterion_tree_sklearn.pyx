# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Lars Buitinck
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

from libc.stdio cimport printf



from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport fabs

import numpy as np
cimport numpy as cnp
cnp.import_array()

from numpy.math cimport INFINITY
from scipy.special.cython_special cimport xlogy

from sklearn.tree._utils cimport log
from sklearn.tree._utils cimport WeightedMedianCalculator

from sklearn.tree._utils cimport log

from sklearn.tree._criterion cimport Criterion

from sklearn.tree._tree cimport DTYPE_t          # Type of X
from sklearn.tree._tree cimport DOUBLE_t         # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t           # Type for indices and counters
from sklearn.tree._tree cimport INT32_t          # Signed 32 bit integer
from sklearn.tree._tree cimport UINT32_t         # Unsigned 32 bit integer


cdef class DiscretizationCriterion(Criterion):
    """Abstract criterion for classification."""

    def __cinit__(self, SIZE_t n_outputs,
                  cnp.ndarray[SIZE_t, ndim=1] n_classes):
        """Initialize attributes for this criterion.

        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets, the dimensionality of the prediction
        n_classes : numpy.ndarray, dtype=SIZE_t
            The number of unique classes in each target
        """
        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = 0
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.n_classes = np.empty(n_outputs, dtype=np.intp)

        cdef SIZE_t k = 0
        cdef SIZE_t max_n_classes = 0

        # For each target, set the number of unique classes in that target,
        # and also compute the maximal stride of all targets
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

            if n_classes[k] > max_n_classes:
                max_n_classes = n_classes[k]

        self.max_n_classes = max_n_classes

        # Count labels for each output
        self.sum_total = np.zeros((n_outputs, max_n_classes), dtype=np.float64)
        self.sum_left = np.zeros((n_outputs, max_n_classes), dtype=np.float64)
        self.sum_right = np.zeros((n_outputs, max_n_classes), dtype=np.float64)


        self.num_dem_result = np.zeros(2, dtype=np.float64)


        self.sum_kplus = 0
        self.sum_kplus_left = 0
        self.sum_kplus_right = 0

        

        self.sum_kminus = 0
        self.sum_kminus_left = 0
        self.sum_kminus_right = 0


    def __reduce__(self):
        return (type(self),
                (self.sum_total, "a", self.n_outputs, np.asarray(self.n_classes)), self.__getstate__())



    cdef int init(
        self,
        const DOUBLE_t[:, ::1] y,
        const DOUBLE_t[:] sample_weight,
        double weighted_n_samples,
        SIZE_t* samples,
        SIZE_t start,
        SIZE_t end
    ) nogil except -1:
        """Initialize the criterion.

        This initializes the criterion at node samples[start:end] and children
        samples[start:start] and samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : ndarray, dtype=DOUBLE_t
            The target stored as a buffer for memory efficiency.
        sample_weight : ndarray, dtype=DOUBLE_t
            The weight of each sample stored as a Cython memoryview.
        weighted_n_samples : double
            The total weight of all samples
        samples : array-like, dtype=SIZE_t
            A mask on the samples, showing which ones we want to use
        start : SIZE_t
            The first sample to use in the mask
        end : SIZE_t
            The last sample to use in the mask
        """
        self.y = y
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef SIZE_t c
        cdef SIZE_t value
        cdef DOUBLE_t w = 1.0

        

        for k in range(self.n_outputs):
            memset(&self.sum_total[k, 0], 0, self.n_classes[k] * sizeof(double))
        
        self.sum_kplus = 0
        self.sum_kminus = 0

        for p in range(start, end):
            i = samples[p]

            # w is originally set to be 1.0, meaning that if no sample weights
            # are given, the default weight of each sample is 1.0.
            if sample_weight is not None:
                w = sample_weight[i]

            # Count weighted class frequency for each target
            for k in range(self.n_outputs):
                c = <SIZE_t> self.y[i, k]
                self.sum_total[k, c] += w

            k = 0
            c = <SIZE_t> self.y[i, k]
            self.sum_kplus += c

            k = 1
            c = <SIZE_t> self.y[i, k]
            self.sum_kminus += c


            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.pos = self.start

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        cdef SIZE_t k

        for k in range(self.n_outputs):
            memset(&self.sum_left[k, 0], 0, self.n_classes[k] * sizeof(double))
            memcpy(&self.sum_right[k, 0], &self.sum_total[k, 0], self.n_classes[k] * sizeof(double))

        self.sum_kplus_left = 0
        self.sum_kplus_right = self.sum_kplus
        self.sum_kminus_left = 0
        self.sum_kminus_right = self.sum_kminus
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.pos = self.end

        self.weighted_n_left = self.weighted_n_node_samples
        self.weighted_n_right = 0.0
        cdef SIZE_t k

        for k in range(self.n_outputs):
            memset(&self.sum_right[k, 0], 0, self.n_classes[k] * sizeof(double))
            memcpy(&self.sum_left[k, 0],  &self.sum_total[k, 0], self.n_classes[k] * sizeof(double))
        
        self.sum_kplus_left = self.sum_kplus
        self.sum_kplus_right = 0
        self.sum_kminus_left = self.sum_kminus
        self.sum_kminus_right = 0
        
        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left child.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        new_pos : SIZE_t
            The new ending position for which to move samples from the right
            child to the left child.
        """
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end

        cdef SIZE_t* samples = self.samples
        cdef const DOUBLE_t[:] sample_weight = self.sample_weight

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef SIZE_t c
        cdef SIZE_t idx
        cdef SIZE_t value
        cdef DOUBLE_t w = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #   sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_po.
        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    self.sum_left[k, <SIZE_t> self.y[i, k]] += w

                idx = 0
                value = <SIZE_t> self.y[i, idx]
                self.sum_kplus_left += value

                idx = 1
                value = <SIZE_t> self.y[i, idx]
                self.sum_kminus_left += value

                self.weighted_n_left += w

                

        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    self.sum_left[k, <SIZE_t> self.y[i, k]] -= w


                idx = 0
                value = <SIZE_t> self.y[i, idx]
                self.sum_kplus_left -= value
                
                idx = 1
                value = <SIZE_t> self.y[i, idx]
                self.sum_kminus_left -= value

                self.weighted_n_left -= w

        # Update right part statistics
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                self.sum_right[k, c] = self.sum_total[k, c] - self.sum_left[k, c]


        self.sum_kplus_right = self.sum_kplus - self.sum_kplus_left

        self.sum_kminus_right = self.sum_kminus - self.sum_kminus_left

        self.pos = new_pos
        return 0

    

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] and save it into dest.

        Parameters
        ----------
        dest : double pointer
            The memory address which we will save the node value into.
        """
        
        

        
        dest[0] = <double> self.sum_kplus
        dest[1] = <double>  (self.sum_kminus+self.sum_kplus)
        

cdef class Entropy_MODIFIED(DiscretizationCriterion):

    r"""Cross Entropy impurity criterion.

    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let

        count_k = 1 / Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The cross-entropy is then defined as

        cross-entropy = -\sum_{k=0}^{K-1} count_k log(count_k)
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node.

        Evaluate the cross-entropy criterion as impurity of the current node,
        i.e. the impurity of samples[start:end]. The smaller the impurity the
        better.
        """
        cdef double entropy = 0.0
        cdef double entropy2 = 0.0
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c




        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                count_k = self.sum_total[k, c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_node_samples
                    entropy -= count_k * log(count_k)


        if self.sum_kplus>0 and self.sum_kminus+self.sum_kplus>0:
        
            fo = self.sum_kplus/(self.sum_kminus+self.sum_kplus)
            if fo>0:
                entropy2 =  - fo * log(fo) - (1-fo) * log(1-fo)
                
        return entropy2 #/ self.n_outputs

    
    def __reduce__(self):
        
        return (type(self),
                (
                    np.asarray(self.sum_total), "a", self.n_outputs, np.asarray(self.n_classes)), self.__getstate__(), 
                    #"y", np.array(self.y), \
                  "sum_left", np.array(self.sum_left), "sum_right", np.array(self.sum_right), \
                  "kplus",  self.sum_kplus, 'kminus', self.sum_kminus,\
                  "left", self.sum_kplus_left, self.sum_kminus_left, \
                  'right',self.sum_kplus_right, self.sum_kminus_right,
                  'impurity', self.node_impurity())




    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).

        Parameters
        ----------
        impurity_left : double pointer
            The memory address to save the impurity of the left node
        impurity_right : double pointer
            The memory address to save the impurity of the right node
        """
        cdef double entropy_left = 0.0
        cdef double entropy_right = 0.0
        cdef double entropy_left_2 = 0.0
        cdef double entropy_right_2 = 0.0
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                count_k = self.sum_left[k, c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_left
                    entropy_left -= count_k * log(count_k)

                count_k = self.sum_right[k, c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_right
                    entropy_right -= count_k * log(count_k)


        if self.sum_kplus_left>0:
            fo_left = self.sum_kplus_left /(self.sum_kminus_left + self.sum_kplus_left)
        
            entropy_left_2 =  - fo_left * log(fo_left) -  (1-fo_left) * log(1-fo_left)

        if self.sum_kplus_right>0:
            fo_right = self.sum_kplus_right/(self.sum_kminus_right+ self.sum_kplus_right)
        
            entropy_right_2 =  - fo_right * log(fo_right) -  (1-fo_right) * log(1-fo_right)

        impurity_left[0] = entropy_left_2 #/ self.n_outputs
        impurity_right[0] = entropy_right_2 #/ self.n_outputs

