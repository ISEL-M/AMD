# to use accented characters in the code
# -*- coding: cp1252 -*-
# ===============================
# author: Paulo Trigo Silva (PTS)
# version: v06
# [new] generateBasket; returns all_itemset
# [new] generateDataFile_basket
# [new] generateDataFile_tab
# Python3 \ Orange3
# ===============================



#_______________________________________________________________________________
# Modules to Evaluate
import csv
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
# define the "normalize" process that applies to each string
def normalizeString( aString ):
    # substitute "=" symbol because it is used in the basket format
    symbolToReplace = "="
    #symbolNew = "|" #this substitution does not work in Orange3
    symbolNew = "+"
    aString = aString.replace( symbolToReplace, symbolNew )
    
    # eliminate spaces (white characters)
    symbolToReplace = " "
    symbolNew = ""
    aString = aString.replace( symbolToReplace, symbolNew )

    # eliminate quotes (" character)
    symbolToReplace = "\""
    symbolNew = "$"
    aString = aString.replace( symbolToReplace, symbolNew )
    
    # eliminate accent characters
    encoding_windows = "iso-8859-1" #"cp1252" #"latin-1" #"latin9" 
    aString = remove_accents( aString, encoding_windows )
    
    # to lower
    aString = aString.lower()
 
    return aString



#_______________________________________________________________________________
# generate the "basket information" from a dataset file with the format:
# TransactioID;ProductID
def generateBasket( fileNameIN ):
    indexTransaction = 0
    indexItem = 1
    basket = {}
    all_itemset = {}
    len_itemset = 0
    #with open( fileNameIN, 'rb' ) as f:
    with open( fileNameIN ) as f:
        reader = csv.reader( f, delimiter=';', quoting=csv.QUOTE_NONE )
        for row in reader:
            #print( row )
            
            transactionID = row[ indexTransaction ]
            #if transactionID not in basket.keys():
            if transactionID not in basket:
                basket[ transactionID ] = {}

            itemID = row[ indexItem ]
            itemID = normalizeString( str( itemID ) )
            
            #if itemID not in basket[ transactionID ].keys():
            if itemID not in basket[ transactionID ]:
                basket[ transactionID ][ itemID ] = 0
            basket[ transactionID ][ itemID ] += 1

            # build an itemset with all items
            # each item is associated with an order relation (len_itemset)
            # the order will be used to generate the basket matrix (.tab) format
            if itemID not in all_itemset:
                all_itemset[ itemID ] = len_itemset
                len_itemset += 1
    return ( basket, all_itemset )



#_______________________________________________________________________________
# generate a dataset file with the ".basket" structure expected by Orange
def generateDataFile_basket( basket, fileNameOUT ):
    with open ( fileNameOUT, mode='wt', encoding='utf-8' ) as f:
        #for transactionID in basket.keys():
        for transactionID in basket:
            line = ""
            for item in basket[transactionID]:
                line+=item
            f.write( line + '\n' )


#_______________________________________________________________________________
# generate a dataset file with the ".tab" structure (basket as a matrix)
def generateDataFile_tab( basket, all_itemset, fileNameOUT ):
    # ordered list of all itemset where the keys are ordered by value
    all_itemset_list = [ k for k in sorted( all_itemset, key=all_itemset.get ) ]

    # ordered list of all transacionID
    all_transactionID_list = list( basket.keys() )
    all_transactionID_list.sort()

    # build .tab file    
    with open ( fileNameOUT, mode='wt', encoding='utf-8' ) as f:
        #_________________
        # build meta-data:
        # (a) name of each feature (item)
        # (b) type of each featura (d = discrete)
        # (c) blank line (with tabs) because there is no class
        #--
        
        # (a):
        line = ""
        for itemName in all_itemset_list:
            if line != "": line += "\t"
            line += str( itemName )
        f.write( line + '\n' )

        # (b):
        line = ""
        for _ in all_itemset_list:
            if line != "": line += "\t"
            line += str( "d" )
        f.write( line + '\n' )

        # (c):
        line = ""
        len_itemset_list = len( all_itemset_list )
        for n in range( len_itemset_list ):
            if n < len_itemset_list-1: line += "\t"
        f.write( line + '\n' )

        #___________________________
        # build data (basket matrix)
        for transactionID in all_transactionID_list:
            # build a list of zeros (default number of items at each transaction)
            line_list = [0] * len_itemset_list
            # set the number of items at each transaction
            for itemID in basket[ transactionID ]:
                itemID_index = all_itemset[ itemID ]
                line_list[itemID_index]+=basket[ transactionID ][ itemID ]
                
            # convert the line_list to a string (elements are separated by tab)
            line = ""
            for number in line_list:
                if line != "": line += "\t"
                line += str( number )
            f.write( line + '\n' )



#_______________________________________________________________________________
# the main of this module (in case this module is imported from another module)
if __name__=="__main__":
    # assumption: the CSV file does not contain the header line
    # (make sure that the export script does not generate the CSV header)
    fIN  = "z_dataset_sample_OUT_expected.txt" #"z_abstract_test.txt" #"z_dataset_sample_OUT.txt"
    fOUT = "zz_dataset_2012_01" #"z_abstract_test"     #"zz_dataset_2012_01"
    print()
    print( ">> 1. Generate Basket structure from CSV file: " + fIN )
    ( basket, all_itemset ) = generateBasket( fIN )

    fOUT_basket = fOUT + ".basket"
    fOUT_tab    = fOUT + ".tab"

    print( ">> 2. Generate .basket dataset file: " + fOUT_basket )
    generateDataFile_basket( basket, fOUT_basket )
    
    print( ">> 3. Generate .tab dataset file: " + fOUT_tab )
    generateDataFile_tab( basket, all_itemset, fOUT_tab )
    




