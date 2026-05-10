import numpy as np
import copy

class partition:
    def __init__(self, partition_encoding):
        """
        Initialize the partition from partition_encoding contained in a list of lists of tuples. 
        The tuples reprsent the row indices i and the column indices j of IJ,
        E.g. [[(0,1),(1,1)],[(1,0),(0,0)]] corresponds to the partition {{(1,2),(2,2)},{(2,1),(1,1)}} of 2 x 2.

        Parameters:
            partition_encoding (list): A list of lists of tuples.
        """
        if not isinstance(partition_encoding, list) or not all(isinstance(sublist, list) for sublist in partition_encoding):
            raise ValueError("Input must be a list of lists.")
        if not all(all(isinstance(item, tuple) for item in sublist) for sublist in partition_encoding):
            raise ValueError("Each element in the sublists must be a tuple.")
        self.partition_encoding = partition_encoding  ### Not used
        self.par = self.partition_encoding
        if not self.check_is_valid_partition():
            raise ValueError("The partition does not contain valid tuples or is empty.")
        self.L = len(self.par)

    # def generate_partition_from_list(self):
    #     """
    #     Generate a partition from a list of lists of tuples.

    #     Returns:
    #         list: A list of sets, where each set contains indices that belong to the same subset of the partition.
    #     """
    #     par = []
    #     for subset in self.partition_encoding:
    #         current_partition = set(subset)
    #         par.append(current_partition)
    #     return par

    def check_is_valid_partition(self):
        """
        Check if the partition does not contain duplicate tuples.

        Returns:
            bool: True if the partition is valid, False otherwise.
        """
        all_elements = set()
        for subset in self.par:
            all_elements.update(subset)
        return len(all_elements) == sum(len(subset) for subset in self.par) and len(self.par) > 0

    def get_length(self):
        ''' return the length of the partition '''
        return self.L
    
    def get_partition(self):
        ''' return the partition '''
        return self.par

    def get_element(self,i):
        ''' return the i-th subset of the partition'''
        return self.par[i]
    
    def get_complement(self,l):
        '''returns a list of the elements of the partition without the l-th subset of the partition'''
        if l<0 or l>=self.L:
            raise ValueError("l must be in [0,L-1]")
        complement=[]
        for i in range(self.L):
            if i!=l:
                complement.append(self.par[i])
        return complement 

    # def partition_diff(self, other_partition: partition):
    #     """
    #     Compare the current partition with another partition and return the indices and corresponding
    #     partition that do not coincide with the other partition.

    #     Parameters:
    #         other_partition (list): A list of sets representing another partition. NOT A PARTITION OBJECT!

    #     Returns:
    #         list: indices and subsets of self.par that are not contained in other_partition.
    #     """
    #     if not isinstance(other_partition, list) or not all(isinstance(s, set) for s in other_partition):
    #         raise ValueError("Other partition must be a list of sets.")

    #     current_partition = self.par
    #     diffsets = []
    #     indices=[]

    #     for idx, temp_part in enumerate(current_partition): ## FIXIT: This does not work anymore since self.par is a list of tuples, not sets.
    #         if temp_part not in other_partition:
    #             diffsets.append( temp_part)
    #             indices.append(idx)

    #     return (indices, diffsets)
    

    def partition_intersect_diff(self, other_partition: "partition"):
        """
        Compare the current partition with another partition and return the indices and corresponding
        elements (lists of tuples) of the current partition that are not elements of the other partition.

        Parameters:
            other_partition (partition): Another partition object.

        Returns:
            list of list: indices and subsets of self.par that are not contained in other_partition.
        """
        if not isinstance(other_partition, partition):
            raise ValueError("other_partition must be a partition object.")
        current_partition = self.par
        other_par = other_partition.par
        other_par_set=[]
        for subset in other_par:
            other_par_set.append(set(subset))
        diffsets = []
        indices = []

        for idx, temp_part in enumerate(current_partition):
            if set(temp_part) not in other_par_set:
                diffsets.append(temp_part)
                indices.append(idx)

        return (indices, diffsets)


# part = partition([[tuple(np.random.randint(size=2, high=4, low=1)) for _ in range(3)] for _ in range(3)]) #should throw an error if the partition is not valid
# #part1 = copy.deepcopy(part.get_partition())

# # print(part.get_partition())
# # print(part.get_complement(0))
# # print(part.check_is_valid_partition())
# p1=partition([[(1,2),(2,1)],[(2,2),(1,1)
#                              ]])
# print(p1.check_is_valid_partition())
# p2=partition([[(2,1),(1,2)],[(2,2)],[(1,1)]])
# print(p2.check_is_valid_partition())
# print(p1.partition_intersect_diff(p2))
# print(p2.partition_intersect_diff(p1))
# print(p2.get_complement(2))
# print(p1.partition_intersect_diff(p1))