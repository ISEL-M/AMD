# -*- coding: cp1252 -*-
# ===============================
# author: Paulo Trigo Silva (PTS)
# version: v06
# ===============================
# 
# "Singleton" possible implementation
# - described in: PEP 318
# (PEP = "Python Enhancement Proposals")



#______________________________________________________________________________
# this function is defined just once
def singleton( cls ):
    instances = {}
    print( "--> applying-the-decorator <--" )
    def getInstance():
       if cls not in instances:
          instances[ cls ] = cls()
       return instances[ cls ]
    return getInstance
#______________________________________________________________________________
