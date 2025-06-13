/* db_proving.pl

    Author:        Steve Moyle
    E-mail:        katarapkokid@hotmail.com
    Copyright (c)  2017
    All rights reserved.

@compat Each ODBC datasource connected to must have a compatible schema.

@bug Tabling at runtime is not supported by the tabling library :-(
*/

:- module(db_proving, [
              register_db/2,  % +Name, +DSNorString
              register_db/3,   % +Name, +DSNorString, +ODBCOptions
              unregister_dbs/0,
              register_db_predicate/2, % +PredArity, +SelectString
              register_db_predicate/3, % +PredSlashArity, +SelectString, +Options
              unregister_db_predicate/1, % +PredSlashArity
              unregister_all_db_predicates/0,
              register_db_for_proving/1, %+DBConnectionAlias
              call_with_db/1, % +Goal
              call_with_db/2, % +Goal, +[CallOptions]
              call_using_db/2, % +Goal, +DBConnectionAlias
              call_using_db/3, % +Goal, +DBConnectionAlias, +[CallOptions]
              db_extent/5, %+Goal, +DBConnectionAlias, +CallOptions, -Goals, -Count
              db_extent/4, %       +Goal, +DBConnectionAlias, -[Goals], -Count
              '$acuity_global'/2
          ]).


:- use_module(library(lists)).
:- use_module(library(odbc)).
:- use_module(library(option)).
% :- use_module(library(tabling)). % @tbd  Tabling at runtime is not supported by the tabling library

:- use_module(library(debug)).

% :- debug(bro_db_connect).
% :- debug(fake_dsn).

default_odbc_options([open(once), null(unset)]).  % map '$null$' to 'unset' to match Sam Hicks' encoding


:- dynamic(user:'$acuity_global'/2). %+Name, -Term

/** <module> DB Proving

This library connection to an SQL database via ODBC to access
data in the SQL database as datalog predicates.

The library allows the same predicates to be proved via alternative
sql databases with a compatible schema.

*/


call_with_db(Goal, CallOptions) :-
    (   user:'$acuity_global'(db_for_proving, DBConnectionAlias)
    ->  true
    ;   format('Warning: No db set for proving', []),
        fail
    ),
    call_using_db(Goal, DBConnectionAlias, CallOptions),
    true.

call_with_db(Goal) :-
    call_with_db(Goal, []).


%!  call_using_db(+Goal, +DBConnectionAlias, +[CallOptions])
%
%   call +Goal using a database query to the database opened with handle
%   +DBConnectionAlias.
%   +CallOptions can be:
%          tabling(true) -- make this Goal be memoized (for subsequent
%          faster lookup
%          db_options([DBOptions]) -- where DBPOptions is a list of
%          options that are valid for the ODBC library.


% @tbd  Tabling at runtime is not supported by the tabling library

% Note: To pass options to the ODBC driver call, use a list of options
% wrapped in db_options([List of ODBC options]) as an element of the
% +CallOptions.
% e.g.  CallOptions = [ tabling(true), db_options([null('$null')] ]

call_using_db(Goal, DBConnectionAlias, CallOptions) :-
    register_db_for_proving(DBConnectionAlias),
    functor(Goal, Pred, Arity),
    user:'$acuity_global'(db_predicate, db_predicate(Pred/Arity, SQLQueryString,  DBOptions0)),
    (   memberchk(db_options(DBOptions1), CallOptions) -> append(DBOptions1, DBOptions0, DBOptions) ; DBOptions = []),
%    ( memberchk(tabling(true), CallOptions) -> table(Pred/Arity);   true),  % Tabling at runtime is not supported
    odbc_query(DBConnectionAlias, SQLQueryString, Row, DBOptions),
    Row  =.. [row  | Args],
    Goal =.. [Pred | Args],
    true.

call_using_db(Goal, DBConnectionAlias) :-
    call_using_db(Goal, DBConnectionAlias, []).


db_extent(Goal, DBConnectionAlias, Goals, Count) :-
    db_extent(Goal, DBConnectionAlias, [], Goals, Count).

db_extent(Goal, DBConnectionAlias, CallOptions, Goals, Count) :-
    register_db_for_proving(DBConnectionAlias),
    findall(Goal, call_using_db(Goal, DBConnectionAlias, CallOptions), Goals),
    length(Goals, Count),
    true.



/* register_db_predicate/3 % + PredArity, +SelectString, +OPtions

    Predicates through which one can prove via a connected ODBC
    database.

*/

register_db_predicate(PredSlashArity, SelectString) :-
    register_db_predicate(PredSlashArity, SelectString, []).

register_db_predicate(PredSlashArity, SelectString, Options) :-
    assertion(PredSlashArity =_Pred/Arity),
    assertion(atomic(SelectString)),
    select_query_arity_check(SelectString, Arity),
    assert(user:'$acuity_global'(
               db_predicate, db_predicate(PredSlashArity, SelectString, Options)
           )),
    true.

unregister_db_predicate(PredSlashArity) :-
    assertion(PredSlashArity =_Pred/_Arity),
    retractall('$acuity_global'(db_predicate, db_predicate(PredSlashArity, _SelectString, _Options))),
    true.

unregister_all_db_predicates :-
    retractall(user:'$acuity_global'(db_predicate, _)).

/*  CRUDE CHECK for arity of predicate and likely return arity!

?- atomic_list_concat(L1, 'from', 'select [a], [b], [c] from [d]'), L1 =[H |_], atomic_list_concat(L2, ',', H), length(L2, LL2).

    L1 = ['select[a], [b], [c] ', ' [d]'],
    H = 'select [a], [b], [c] ',
    L2 = ['select[a]', ' [b]', ' [c] '],
    LL2 = 3.

?- select_query_arity_check('select [ts], [uid], [id.orig_h], [id.resp_h] from [conn]', 3).
Warning: SQL Query select [ts], [uid], [id.orig_h],
[id.resp_h] from [conn] does not seem to match the arity required 3.
Mapping registered anyway.
true.

?- select_query_arity_check('select [ts], [uid], [id.orig_h],  [id.resp_h] from [conn]', 4).
 true.


*/
select_query_arity_check(SQLQueryString, Arity) :-
    assertion(atomic(SQLQueryString)),
    (   atomic_list_concat(L1, 'from', SQLQueryString)
    ;   atomic_list_concat(L1, 'FROM', SQLQueryString)   % Note that this is incomplete
    ),
    L1 =[PrefixFrom |_],
    atomic_list_concat(L2, ',', PrefixFrom),
    length(L2, Arity),
    !.
select_query_arity_check(SQLQueryString, Arity) :-
    format('Warning: SQL Query ~w does not seem to match the arity required ~w.~n', [SQLQueryString, Arity]),
    format('Mapping registered anyway.~n', []),
    !.

%!  register_db_for_proving(?DBConnectionAlias) is det
%
%   Set the ODBC database alias through which ODBC queries will be
%   called
register_db_for_proving(DBConnectionAlias) :-
    '$acuity_global'(db_for_proving, DBConnectionAlias),
    assertion(odbc_current_connection(DBConnectionAlias, _DSN)),  % Already exists -- do nothing
    !.
register_db_for_proving(DBConnectionAlias) :-
    retractall(user:'$acuity_global'(db_for_proving, _)),
    assertion(odbc_current_connection(DBConnectionAlias, _DSN)),
    asserta(user:'$acuity_global'(db_for_proving, DBConnectionAlias)).



register_db(Name, RegTerm) :-
    default_odbc_options(DefOptions),
    merge_options([alias(Name)], DefOptions, Options),
    register_db(Name, RegTerm, Options),
    !,
    true.


register_db(Name, dsn(DSN), ODBCOptions) :-
    assertion(ground(DSN)),
    (   odbc_data_source(DSN, _Description)
    ->  debug(bro_db_connect, 'ODBC Options: ~w', [ODBCOptions]),
        assertion(option(alias(Name), ODBCOptions)),
        odbc_connect(DSN, _Conn, ODBCOptions)
    ;   format('No DSN for name ~w~n.  No DB connected.', [DSN]),
        fail
    ).

register_db(Name, sqlite3file(SQLite3FilePath), ODBCOptions) :-
    (   exists_file(SQLite3FilePath)
    ->  format(atom(ConnectionStringTerm), 'Driver=SQLITE3;Database=~w', [SQLite3FilePath]),
        register_db(Name, conn_string(ConnectionStringTerm), ODBCOptions)
    ;   format('No file at ~w~n.  No DB connected.', [SQLite3FilePath]),
        fail
    ).


% we expect that +Name does not exist as a registered ODBC alias.
% If it does, then we must fail.

register_db(Name, conn_string(ConnectionString), ODBCOptions0) :-
    (   option(alias(Name), ODBCOptions0) -> true ; ODBCOptions1 = [alias(Name) | ODBCOptions0]),
        debug(bro_db_connect, 'Connection string: ~w', [ConnectionString]),
        debug(bro_db_connect, 'ODBC Options: ~w', [ODBCOptions1]),
        ODBCOptions = [driver_string(ConnectionString) | ODBCOptions1],
        fake_dsn(ODBCOptions, FakeDSN),
        (   (odbc_current_connection(Name0,  _DSN), Name0 = Name)   % Already open
        ->  format('Connection alias exists ~w~n.  DB Not re-connected.', [Name]),
            !,
            fail
        ;    ( catch(odbc_connect(FakeDSN, Name, ODBCOptions), ThrownTerm, format('Caught ~w~n', [ThrownTerm]))
             ->  true
             ;   format('Connection alias exists ~w~n.  DB not re-connected using ~w.', [Name, ConnectionString]),
                 !,
                 fail
             )
        ).

%fake_dsn(+ODBCOptions, ?FakeDSN)
%
% Generate a fake dsn handle for ODBC connections that do not have
% a DNS in the odbc.ini, but use the options (including the connection
% string instead).
%
% ?
% -fake_dsn([driver_string(Driver=SQLITE3;Database=/home/sam/Documents/SAM/Private/Incubation/CoalesceIntelligence/Swiss_Watch/ASP/EPSRC_Summer_Project/sqlite/test_cw4.sqlite),alias(cw4viastring),null(unset),open(once)',
% FakeDSN).
%
%
% FakeDSN = $dsn(9883046)
%
fake_dsn(ODBCOptions, FakeDSN) :-
    assertion(ground(ODBCOptions)),
    term_hash(ODBCOptions, Hash),
    format(atom(FakeDSN), '$dsn(~w)', [Hash]),
    debug(fake_dsn, 'ODBCOptions: ~w   FakeDSN: ~w', [ODBCOptions, FakeDSN]),
    true.


%!  unregister_dbs/0 is det.
%
%   Remove and close all the ODBC connections that are open.
%
unregister_dbs :-
    odbc_current_connection(Alias, _DSN),
    odbc_disconnect(Alias),
    fail.
unregister_dbs :-
    !.

/* end of db_proving. pl */

