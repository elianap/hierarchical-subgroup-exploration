from sklearn.tree._tree cimport DTYPE_t          # Type of X
from sklearn.tree._tree cimport DOUBLE_t         # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t           # Type for indices and counters
from sklearn.tree._tree cimport INT32_t          # Signed 32 bit integer
from sklearn.tree._tree cimport UINT32_t         # Unsigned 32 bit integer


from sklearn.tree._criterion cimport Criterion


cdef class DiscretizationCriterion(Criterion):
    """Abstract criterion for classification."""

    cdef SIZE_t[::1] n_classes
    cdef SIZE_t max_n_classes

    cdef double kplus   # The sum of the weighted count of each label.
    cdef double kminus   # The sum of the weighted count of each label.

    cdef double[:] num_dem_result
    


    cdef double[:, ::1] sum_total   # The sum of the weighted count of each label.
    cdef double[:, ::1] sum_left    # Same as above, but for the left side of the split
    cdef double[:, ::1] sum_right   # Same as above, but for the right side of the split


    cdef double sum_kplus   # The sum of the weighted count of each label.
    cdef double sum_kminus   # The sum of the weighted count of each label.
    cdef double sum_kplus_left    # Same as above, but for the left side of the split
    cdef double sum_kplus_right   # Same as above, but for the right side of the split
    cdef double sum_kminus_left    # Same as above, but for the left side of the split
    cdef double sum_kminus_right   # Same as above, but for the right side of the split


