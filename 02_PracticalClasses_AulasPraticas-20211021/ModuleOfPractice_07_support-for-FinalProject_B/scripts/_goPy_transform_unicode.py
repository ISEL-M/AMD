# to use accented characters in the code
# -*- coding: cp1252 -*-
# ===============================
# author: Paulo Trigo Silva (PTS)
# version: v04 \ Python3
# ===============================



#_______________________________________________________________________________
# auxiliary file to test the remotion of accents
#_______________________________________________________________________________


#_______________________________________________________________________________
# Modules to Evaluate
import unicodedata


#_______________________________________________________________________________
# convert string to unicode format
# in Python3 all strings are sequences of Unicode characters
# so there is no need to transform strings into Unicode
# therefore this method is deprecated for Python3
##def to_unicode( obj, encoding='utf-8' ):
##    if isinstance( obj, basestring ):
##        if not isinstance( obj, unicode ):
##            obj = unicode( obj, encoding )
##    return obj



#_______________________________________________________________________________
# remove accentes in an unicode string
def remove_accents( aString, encoding='utf-8' ):
    #next instruction is not necessary in Python3
    #aString_unicode = to_unicode( aString, encoding )
    aString_unicode = aString
    nfkd_form = unicodedata.normalize( 'NFKD', aString_unicode )
    only_ascii_form = nfkd_form.encode( 'ascii', 'ignore' ).decode( encoding )
    return only_ascii_form



#_______________________________________________________________________________
# test the "remove_accents" function
#_______________________________________________________________________________
encoding_windows = "iso-8859-1" #"cp1252" #"latin-1" #"latin9" 
#aString = "aàáâãäåÁÀ eéêè iìí oóÒõ uúùü º^º ç ~» ß"
aString = "ü àáâ joão ìí ú inför på fédéral électoral große"
print( aString )

aString = remove_accents( aString, encoding_windows )
print( aString )

