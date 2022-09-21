# -*- coding: cp1252 -*-
# ===============================
# author: Paulo Trigo Silva (PTS)
# version: v06
# ===============================


# use the Singleton in another context
from z01_singleton_usage import C, D

print( "---\\outro-modulo//---" )

c1=C()
c1.f()
c1.inc()

c2=C()
c2.f()


def f():
   print( "---\\another-module::f_a//---" )
   c=C()
   c.f()
   c.inc()
   c.f()

   print( "---\\another-module::f_b//---" )
   
   d=D()
   d.inc()
   d.f()



class H:
   def __init__( self ):
      print( "---\\another-module::class::H::__init__()//---" )
      c=C()
      c.f()
      c.inc()
      c.f()

   def m( self ):
      print( "---\\another-module::class::H::m()//---" )
      c=C()
      c.inc()
      c.f()
        


#_______________________________________________________________________________
def main():
    f()
    H().m()
    print( "---\\another-module::main//---" )
    c=C()
    c.inc()
    c.f()
#_______________________________________________________________________________




#_______________________________________________________________________________
if __name__ == "__main__":
    main()


