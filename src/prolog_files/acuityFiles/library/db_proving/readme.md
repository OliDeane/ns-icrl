= db_proving

A library for proving predicates that have a basis in and ODBC reachable
database (e.g. via SQL).


== Design considerations

1. Enable switch from database to database to compare the coverage of a predicate.
2. SQL syntax should be portable.  (e.g. encapsulate meta data in "[" and "]" where possible).
3. Minimise recomputation of SQL derived predicates (e.g. ustilise the tabling library).


