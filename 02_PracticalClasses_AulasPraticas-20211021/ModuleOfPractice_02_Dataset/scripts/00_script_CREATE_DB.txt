--#############
--# Paulo Trigo
--#############


----------
-- DB name
----------
\set dataBase db_operational
;

-----------------------
-- Remode and Create DB
-----------------------

\echo "Remove Data Base" :dataBase
;

DROP DATABASE IF EXISTS :dataBase
;


\echo "Create Data Base" :dataBase
;

------------------------------------------------------------------------
-- The database is created considering its template
-- if no template is desired just remove the "TEMPLATE = my_db" parameter
------------------------------------------------------------------------
CREATE DATABASE :dataBase TEMPLATE = my_db
;
