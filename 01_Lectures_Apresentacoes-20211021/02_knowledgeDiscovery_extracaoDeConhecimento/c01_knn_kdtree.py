# Para usar caracteres acentuados no codigo
# -*- coding: cp1252 -*-
# ====================================
__author__ = "Paulo Trigo Silva (PTS)"
__all__ = [ "kNN_kdTree_AMD_teorica" ]
# ====================================



#_______________________________________________________________________________
# Os Modulos que devem ser avaliados
import math



#_______________________________________________________________________________
# Distance Functions (Metrics)
def distance_euclidean( exampleA, exampleB ):
   assert len( exampleA ) == len( exampleB )
   return math.sqrt( sum( ( itemA - itemB )**2 for ( itemA, itemB ) in zip( exampleA, exampleB ) ) )


def distance_square( exampleA, exampleB ):
   assert len( exampleA ) == len( exampleB )
   return sum( ( itemA - itemB )**2 for ( itemA, itemB ) in zip( exampleA, exampleB ) )


def distance_manhattan( exampleA, exampleB ):
   assert len( exampleA ) == len( exampleB )
   return sum( abs( itemA - itemB ) for ( itemA, itemB ) in zip( exampleA, exampleB ) )


def distance_hamming( exampleA, exampleB ):
   assert len( exampleA ) == len( exampleB )
   return sum( itemA != itemB for ( itemA, itemB ) in zip( exampleA, exampleB ) )

"""
diff: to deal with both nominal end numeric values
"""
def diff( valueA, valueB ):
   pass
   # <COMPLETAR>



#_______________________________________________________________________________
# Choose Dimension - strategy to select the dimension at each node
def choose_dimension_fromDepth( point_list, depth ):
   # choose dimension based on depth
   # (so that dimension cycles through all valid values)
   # assumes all points have the same dimension
   return depth % len( point_list[ 0 ] )



#_______________________________________________________________________________
# Choose Index - strategy to select the splitting data index at each node
def choose_index_median( point_list ):
   # choose the index that corresponds to the median
   return len( point_list ) / 2



#_______________________________________________________________________________
# TreeNode (a general tree node structure)
class TreeNode():
   def __init__( self, data, children ):
      self.__data = data
      # a list with the root node (TreeNode) of the descending trees
      self.__children = children 
   
   def is_leaf( self ):
      for item in self.children:
         if item <> None: return False
      return True

   @property
   def data( self ): return self.__data
   @data.setter
   def data( self, value ): self.__data = value
   @property
   def children( self ): return self.__children



#_______________________________________________________________________________
# KDTreeNode (the binary kd-tree node with "point" data and "dimension")
class KDTreeNode( TreeNode ):
   def __init__( self, point, dimension, left, right ):
      TreeNode.__init__( self, point, [ left, right ] )
      self.__dimension = dimension

   def toList( self ):
      if not self.left and not self.right: return [ [ self.point, str( self.dimension ) ], None, None ]
      if not self.left: return [ [ self.point, str( self.dimension ) ], None, self.right.toList() ]
      if not self.right: return [ [ self.point, str( self.dimension ) ], self.left.toList(), None ]
      return [ [ self.point, str( self.dimension ) ], self.left.toList(), self.right.toList() ]
      
   @property
   def dimension( self ): return self.__dimension
   @property
   def point( self ): return self.data
   @property
   def left( self ): return self.children[ 0 ]
   @property
   def right( self ): return self.children[ 1 ]



#_______________________________________________________________________________
# KDTree - builds the KDTree from data
class KDTree():
   def __init__( self, data,
                 choose_dimension=choose_dimension_fromDepth,
                 choose_index=choose_index_median ):
      self.__choose_dimension = choose_dimension
      self.__choose_index = choose_index
      self.__root = self.__build( data, depth=0 )
  
   def __build( self, point_list, depth ):
      if not point_list: return None
      # choose the dimension to split the tree
      dimensionSplit = self.choose_dimension( point_list, depth )

      # "sort" modifies the list in-place (and returns None)
      # "key" is a function with a single argument; the return is used for sorting
      point_list.sort( key = lambda point: point[ dimensionSplit ] )

      # choose the (data) index that splits the data in two subtrees
      indexSplit = self.choose_index( point_list )

      # create node and
      # recursively construct subtrees
      #_________________
      node = KDTreeNode(
         point = point_list[ indexSplit ],
         dimension = dimensionSplit,
         left = self.__build( point_list[ 0:indexSplit ], depth + 1 ),
         right = self.__build( point_list[ indexSplit+1: ], depth + 1 ) )
      return node

   def pprint( self ):
      tree_list = self.root.toList()
      node = tree_list[ 0 ]
      left = tree_list[ 1 ]
      right = tree_list[ 2 ]
      self.__pprint( node, left, right, depth=0, label="" )

   def __pprint( self, node, left, right, depth, label ):
      prefix = " "*depth*2 + label
      if not node: info = "[]"
      else: info = str( node[ 0 ] ) + " d=" + str( node[ 1 ] )
      print prefix + info
      if not node: return
      if not left and right: self.__pprint( None, None, None, depth + 1, "/" )
      if left: self.__pprint( left[ 0 ], left[ 1 ], left[ 2 ], depth + 1, "/" )
      if not right and left: self.__pprint( None, None, None, depth + 1, "\\" )
      if right: self.__pprint( right[ 0 ], right[ 1 ], right[ 2 ], depth + 1, "\\" )

   @property
   def root( self ): return self.__root
   @property
   def choose_dimension( self ): return self.__choose_dimension
   @property
   def choose_index( self ): return self.__choose_index



#_______________________________________________________________________________
# NearestNeighbours - keeps memory of (point, distance) tuples
# (internal structure used in nearest-neighbours search)
class NearestNeighbours():
   def __init__( self, point_query, k ):
      self.__point_query = point_query
      self.__k = k
      self.__current_best = []

   def add_ifBetter( self, point, distance_function ):
      distance = distance_function( point, self.__point_query )
      # run through current_best, try to find appropriate place
      for i, e in enumerate( self.__current_best ):
         # all neighbours found, this one is farther, so return
         if i == self.__k: return
         if e[ 1 ] > distance:
            self.__current_best.insert( i, ( point, distance ) )
            return
      # otherwise, append the point and its distance (to the point_query) to the end
      self.__current_best.append( ( point, distance ) )

   @property
   def largest_distance( self ):
      if self.__k >= len( self.__current_best ): return self.__current_best[ -1 ][ 1 ]
      return self.__current_best[ self.__k - 1 ][ 1 ]

   @property
   def best_k_neighbours( self ):
      return [ element[ 0 ] for element in self.__current_best[ :self.__k ] ]



#_______________________________________________________________________________
# KNN - searches the k nearest neighbours using a kdTree strucuture
class KNN():
   def __init__( self, kdTree, distance_function=distance_euclidean ):
      self.__kdTree = kdTree
      self.__distance_function = distance_function
      self.statistics = {}

   """
   point_query = the point to search for k neighbours
   k = the number of neighbours to search for
   """
   def query( self, point_query, k=1 ):
      self.statistics = { 'nodes_visited': 0, 'far_search': 0, 'leafs_reached': 0 }
      if self.__kdTree.root == None: result = []
      neighbours = NearestNeighbours( point_query, k )
      self.__search( self.__kdTree.root, point_query, k, best_neighbours=neighbours )
      result = neighbours.best_k_neighbours
      return result


   def __search( self, node, point_query, k, best_neighbours ):
      if node == None: return
      self.statistics[ 'nodes_visited' ] += 1
      # if leaf, add to current best neighbours
      # (add if better than the worst or if not enough neighbours yet)
      if node.is_leaf():
         self.statistics[ 'leafs_reached' ] += 1
         best_neighbours.add_ifBetter( node.point, self.distance_function )
         return
      # if internal node, get the dimension to split the tree
      # (the dimension was stored in the node durint kdTree construction)
      dimensionSplit = node.dimension
      near_subtree = None
      far_subtree = None
      # compare point_query and point of current node in selected dimension
      # and decide which subtree is the "near" and which is the "far"
      if point_query[ dimensionSplit ] < node.point[ dimensionSplit ]:
         near_subtree = node.left
         far_subtree = node.right
      else:
         near_subtree = node.right
         far_subtree = node.left
      # until a leaf is found,
      # recursively search through the "near" subtree
      #__________________________________________________________
      self.__search( near_subtree, point_query, k, best_neighbours )

      # while unwinding the recursion, check if:
      # current node is closer to query point than the current best,
      # (until k points have been found, search radius is infinity)
      best_neighbours.add_ifBetter( node.point, self.distance_function )
      # check whether there could be any points on the other side of the
      # splitting plane that are closer to the query point than the current best
      point_component = [ node.point[ dimensionSplit ] ]
      point_query_component = [ point_query[ dimensionSplit ] ]
      component_distance = self.distance_function( point_component, point_query_component )
      # the case where we also traverse the "far" subtree subtree
      if component_distance < best_neighbours.largest_distance:
         self.statistics[ 'far_search' ] += 1
         # until a leaf is found,
         # recursively search through the "far" subtree
         #____________________________________________________________
         self.__search( far_subtree, point_query, k, best_neighbours )
      return

   @property
   def distance_function( self ): return self.__distance_function



#_______________________________________________________________________________
# A basic linear kNN search
def kNN_linearSearch( point_list, point_query, k, distance_function=distance_euclidean ):
   best_neighbours = NearestNeighbours( point_query, k )
   for point in point_list: best_neighbours.add_ifBetter( point, distance_function )
   result = best_neighbours.best_k_neighbours
   return result



#_______________________________________________________________________________
# Some utility functions
def my_print( aStr ):
   separator = lambda x: "_" * len( x )
   print separator( aStr )
   print aStr



#_______________________________________________________________________________
# O "main" deste modulo (caso o modulo nao seja carregado de outro modulo)
if __name__=="__main__":
   d = [ (6, 8), (2, 6), (5, 6), (3, 5), (4, 2), (2, 1) ]
   point = (3, 2)
   #d = [ (3, 4), (4, 1), (6, 7), (2, 2), (8.5, 5), (3, 8), (7, 4.5), (5, 7.5) ]
   #point = (3, 2)
   #d = [ (3, "d"), (4, "a"), (6, "g"), (2, "b"), (8.5, "e"), (3, "h"), (7, "e"), (5, "h") ]
   #point = (3, "b")
   #d = [ (3, "d", 4), (4, "a", 6), (6, "g", 2), (2, "b", 8.5) ]
   #point = (3, "b", 6)
   tree = KDTree( d )
   
   print
   my_print( "KD-Tree (na forma de lista):" )
   print tree.root.toList()
   
   print
   my_print( "KD-Tree (na forma de árvore):" )
   tree.pprint()

   knn = KNN( tree, distance_euclidean )
   print
   print
   k = 1
   nearest = knn.query( point, k )
   my_print( "kNN para instância: " + str( point ) + " com k = " + str( k )  )
   print nearest
   my_print( "kNN com KD_Tree - algumas estatísticas:" )
   print knn.statistics

   print
   print
   k = 2
   nearest = knn.query( point, k )
   my_print( "kNN para instância: " + str( point ) + " com k = " + str( k )  )
   print nearest
   my_print( "kNN com KD_Tree - algumas estatísticas:" )
   print knn.statistics
   
   print
   my_print( "..::SEM USAR A KD-Tree::.." )
   print kNN_linearSearch( d, point, k=2 )


   d = [ (-0.1,), (0.7,), (1.0,), (1.6,), (2.0,), (2.5,), (3.2,), (3.5,), (4.1,), (4.9,) ]
   tree = KDTree( d )
   print
   my_print( "KD-Tree (na forma de árvore):" )
   tree.pprint()


