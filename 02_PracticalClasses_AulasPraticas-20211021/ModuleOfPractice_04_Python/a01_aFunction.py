# -*- coding: cp1252 -*-
# ===============================
# author: Paulo Trigo Silva (PTS)
# version: v06
# ===============================



def f( a_list ):
    
   a_list.append("quit").append("exit")
   a_list = [ str( x ).upper() for x in a_list ]
   while True:
      v = input( '? > ' )
      if v.upper() in a_list: break
      #if v.upper() in a_list or v == 'quit' or v == 'exit': break
      print( v.upper() )


if __name__ == "__main__":
   a_list = [ 'o', 1, 33, 'hello' ]
   f( a_list )



