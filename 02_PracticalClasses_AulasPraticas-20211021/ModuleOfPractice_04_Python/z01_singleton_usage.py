# -*- coding: cp1252 -*-
# ===============================
# author: Paulo Trigo Silva (PTS)
# version: v06
# ===============================



#______________________________________________________________________________
# The modules to import
from x_singleton import singleton



# C is a Singleton
# so it is decoracted with "singleton"
@singleton
class C:
    def __init__( self ):
       self.c = 1

    def f( self ):
       print( self.c )

    def inc( self ):
       self.c += 1


c1=C()
c1.f()
c1.inc()

c2=C()
c2.f()

print( "---//\\---" )



# D is a Singleton
# so it is decoracted with "singleton"
@singleton
class D:
    def __init__( self ):
       self.d = 1

    def f( self ):
       print( self.d )

    def inc( self ):
       self.d += 1



d1=D()
d1.f()
d1.inc()

d2=D()
d2.f()

print( "---//\\---" )
