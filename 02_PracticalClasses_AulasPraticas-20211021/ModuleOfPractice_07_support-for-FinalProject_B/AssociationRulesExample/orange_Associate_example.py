# to use accented characters in the code
# -*- coding: cp1252 -*-
# ===============================
# author: Paulo Trigo Silva (PTS)
# version: v02 \ Python3
# ===============================


#_______________________________________________________________________________
# example - usage of the "associate" module in Orange3
#_______________________________________________________________________________



#_______________________________________________________________________________
# Modules to Evaluate
import Orange
from orangecontrib.associate.fpgrowth import *


#_______________________________________________________________________________
# read the dataset (basket format)
Xbasket = Orange.data.Table( "market-basket.basket" )
print()
print( "The original basket file:" )
print( Xbasket )


#_______________________________________________________________________________
# make the OneHot transform in order to apply the "associate" methods
# cf., http://orange3-associate.readthedocs.io/en/latest/scripting.html
X, mapping = OneHot.encode( Xbasket )

# visualize the OneHot transform
# (notice that item multiplicity was ignored)
print()
print( "The transformed basket file:" )
for x in X: print( x )


#_______________________________________________________________________________
# search for the set of "frequent itemsets"
# "frequent_itemsets" returns an "iterator" (or generator)
# therefore if must be "consumed" after being invoked
# (the frequent_itemsets" function uses internally a "yield" function)
support = 2
setOf_itemset = frequent_itemsets( X, support )

print()
print( "The set-of itemsets:" )
# here the "setOf_itemset" is consumed during the print loop
for itemset in setOf_itemset: print( itemset )

# now if we print the "setOf_itemset" we see that it is empty
# (this is just for illustrative purposes)
print( "Again, the set-of itemsets:" )
for itemset in setOf_itemset: print( itemset )


#_______________________________________________________________________________
# the "setOf_itemset" has already been "consumed" (in the previous "print" loop)
# so we must invoke it again
confidence = 0.6
setOf_itemset = frequent_itemsets( X, support )
# the next line "consumes" all the "setOf_itemset" and builds a dictionary from it
# (this way we can use it for future operations)
dict_setOf_itemset = dict( setOf_itemset )
setOf_rule = association_rules( dict_setOf_itemset, confidence )

# the next line "consumes" all the "setOf_rule" and builds a list from it
# (this way we can use it for future operations)
list_setOf_rule = list( setOf_rule )

print()
print( "The set-of rules:" )
for rule in list_setOf_rule: print( rule )


#_______________________________________________________________________________
# decode the itemsets back to their original values
print()
print( "Decoded frequent-itemsets:" )
for itemset in dict_setOf_itemset.keys():
   supp = dict_setOf_itemset[itemset]
   decoded_itemset = [ var.name for _, var, _ in OneHot.decode( itemset, Xbasket, mapping ) ]
   tuple_itemset = ( decoded_itemset, supp )
   print( tuple_itemset )


#_______________________________________________________________________________
# decode the setOf-rules back to their original values
print()
print( "Decoded setOf-rules:" )
for rule in list_setOf_rule:
   LHS, RHS, supp, conf = rule
   decoded_LHS = [ var.name for _, var, _ in OneHot.decode( LHS, Xbasket, mapping ) ]
   decoded_RHS = [ var.name for _, var, _ in OneHot.decode( RHS, Xbasket, mapping ) ]
   tuple_rule = ( decoded_LHS, decoded_RHS, supp, conf )
   print( tuple_rule )



