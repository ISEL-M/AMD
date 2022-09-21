# -*- coding: cp1252 -*-
# ===============================
# author: Paulo Trigo Silva (PTS)
# version: v06
# ===============================



print( "\n----- One Example -----" )

def f( v="oneParameter" ):
   def g(): return v
   return g

x = f()
print( x )
print( x() )



print( "\n----- Another Example -----" )

def f( v1="oneParameter" ):
   def g( v2="anotherParameter" ):
      def h(): return v1 + "//" + v2
      return h
   return g

x = f()
print( x )
print( x() )
print( x()() )
print( f( "http:" )( "www" )() )



#__________________________________________________________________________
# Preliminary experiments (to support the understanding of @decorator)
# a function that returns another function; remarks:
# - the variable v exists in the scope of the local functions
# INPORTANT:
# this construction enables to formulate the notion of "class", where:
# - c, is a class
# - s_m, is the scope of the methods
# - m, is a method
#__________________________________________________________________________
def c( v=0 ):
   def s_m( x ):
      if x == 1:
         def m( y ): return y**2 + v
      elif x == 2:
         def m( y ): return "< " + str(y) + " >" + "$ " + str(v) + " $"
      else:
         def m(): return "---"
      return m
   return s_m

print( "\nPreliminary experiments (function that returns a function):" )
o = c()
# note that o is a function
print( o( 1 ) )
print()
o1 = o( 1 )
print( o1( 100 ) )
print( o( 2 )( "ola" ) )
print( o( 3 )() )
# OR
print( c()( 1 )( 100 ) )
print( c( "www" )( 2 )( "ola" ) )
print( c()( 3 )() )
#__________________________________________________________________________



#__________________________________________________________________________
# seja:
# f( g ( h( x ) ) )
# if we intend to define h as the result of this composition
# i.e., if we intend to do:
# h = f( g ( h ) )
# IMPORTANT: note that h is a function (f o g)
# and not the result of invoking g and f
# then we write
# @f
# @g
# def h(): ...
#
# and, in this case we say that f and g are "decorators" of h
#__________________________________________________________________________

# Example:
def f( fn ):
   def local():
      s = "< " + fn() + " >"
      return s
   return local

def g( fn ):
   def local():
      s = "# " + fn() + " #"
      return s
   return local
   
print( "\nVersion with @decorator syntax:" )
@f
@g
def h():
   s = "h"
   return s

print( h() )

print( "\nVersion with direct (explicit) definition:" )
def h():
   s = "h"
   return s

h = f( g ( h ) )
print( h() )
