% -*- Prolog -*-

/* db_proving.plt
 *
 *  Tests for db_proving.pl
 *
 *  ?- ['db_proving.plt'].
 *  ?- run_tests.
 *
 *  */

:- use_module(library(plunit)).

:- begin_tests(db_proving).

:- use_module('db_proving.pl').

:- use_module(library(odbc)).

:- use_module(library(debug)).
% :- debug(registration).

/*
un_register(Conn) :-
      (   call_cleanup(odbc_current_connection(Conn0, _DSN), _, true)),
      (   Conn0 = Conn
      ->  odbc_disconnect(Conn)
      ;   debug(registration, '~nConnection ~w was not registered', [Conn])
      ),
      !.
*/
un_register(Alias) :-
     catch(odbc_disconnect(Alias), E, print_message(error, E)).

test(register_db_via_dsn, [nondet]) :-
      (   un_register(cw4viadsn)
      ;   debug(registration, '~nConnection cw4viadsn was not registered', [])
      ),
      odbc_data_source(cw4sqlite, Description),
      debug(registration, '~nDSN [cw4sqlite] has description ~w', [Description]),
      register_db(cw4viadsn, dsn(cw4sqlite)),
      odbc_current_connection(cw4viadsn, Conn),
      Conn = cw4sqlite,
      odbc_current_table(cw4viadsn, conn).

test(register_db_via_file, []) :-
      un_register(cw4viafile),
%      register_db(cw4viafile,
%      sqlite3file('/home/sam/Documents/SAM/Private/Incubation/CoalesceIntelligence/Swiss_Watch/ASP/EPSRC_Summer_Project/sqlite/test_cw4.sqlite')),
      absolute_file_name('./tools/sqlite3/test_cw4.sqlite', FullPath, []),
      exists_file(FullPath),
      register_db(cw4viafile, sqlite3file(FullPath)),
      odbc_current_connection(cw4viafile,  Conn),
      debug(registration, '~nCurrent connection cw4viafile ~w', [Conn]),
      once(odbc_current_table(cw4viafile, conn)). % remove non-det  warning


test(register_db_via_string, []) :-
      un_register(cw4viastring ),
      register_db(cw4viastring,
                 conn_string('Driver=SQLITE3;Database=/home/sam/Documents/SAM/Private/Incubation/CoalesceIntelligence/Swiss_Watch/ASP/EPSRC_Summer_Project/sqlite/test_cw4.sqlite')
                 ),
      odbc_current_connection(cw4viastring,  Conn),
      debug(registration, '~nCurrent connection cw4viastring ~w', [Conn]),
      once(odbc_current_table(cw4viastring, conn)). % remove non-det  warning


test(register_db_predicate_with_options, []) :-
      ResultSpec = conn/3,
      SelectString = 'select [ts], [uid], [id.orig_h], [id.resp_h] from [conn]',
      unregister_db_predicate(ResultSpec),
      register_db_predicate(ResultSpec, SelectString, []),
      '$acuity_global'(db_predicate, db_predicate(PredSlashArity, SelectString1, Options)),
      ground(PredSlashArity),
      PredSlashArity = ResultSpec,
      ground(SelectString1),
      SelectString1 = SelectString,
      Options = [],
      true.


/*
?- findall(C->D, call_with_db(conn(_A, _B, C, D), cw4viafile), As), length(As, LAs).
As = [('192.168.122.163'->'103.21.59.9'),  ('192.168.122.163'->'173.237.136.250'),  ('192.168.122.163'->'195.208.1.122'),  ('192.168.122.163'->'192.168.122.2'),  ('192.168.122.163'->'192.168.122.2'),  ('192.168.122.163'->'192.168.122.2'),  ('192.168.122.163'->'103.21.59.9'),  ('192.168.122.163'->'173.237.136.250'),  (... -> ...)|...],
LAs = 30.

*/
test(call_using_db_conn4_findall, []) :-
      ResultSpec = conn/4,
      SelectString = 'select [ts], [uid], [id.orig_h], [id.resp_h] from [conn]',
      unregister_db_predicate(ResultSpec),
      register_db_predicate(ResultSpec, SelectString, []),
      findall(C->D, call_using_db(conn(_A, _B, C, D), cw4viafile), As),
      length(As, LAs),
      LAs = 30.

test(call_using_db_conn4_bound, []) :-
      ResultSpec = conn/4,
      SelectString = 'select [ts], [uid], [id.orig_h], [id.resp_h] from [conn]',
      unregister_db_predicate(ResultSpec),
      register_db_predicate(ResultSpec, SelectString, []),
      A = 1446728687.0558,
      B = 'CoPfcP251c0hXRko86',
      C = '192.168.122.163' ,
      D = '192.168.122.2',
      once(call_using_db(conn(A, B, C, D), cw4viafile)).


test(call_with_db, []) :-
      register_db_for_proving(cw4viafile),
      A = 1446728687.0558,
      B = 'CoPfcP251c0hXRko86',
      C = '192.168.122.163' ,
      D = '192.168.122.2',
      once(call_with_db(conn(A, B, C, D))).
/*
?- db_extent(conn(A, B, C, '103.21.59.9'), cw4viafile, Gs, L).
Gs = [conn(1446728684.76495, 'CiT3vV2lnFWQC0NIAl', '192.168.122.163', '103.21.59.9'), conn(1446728689.17154, 'CSJUFE9sxOMgQkSSl', '192.168.122.163', '103.21.59.9'), conn(1446728712.45098, 'CCJwMV28pNkahEhhTg', '192.168.122.163', '103.21.59.9')],
L = 3.
*/


test(db_extent_conn4, []) :-
      db_extent(conn(_A, _B, _C, '103.21.59.9'), cw4viafile, Gs, L),
      Gs = [conn(1446728684.76495, 'CiT3vV2lnFWQC0NIAl', '192.168.122.163', '103.21.59.9'),
            conn(1446728689.17154, 'CSJUFE9sxOMgQkSSl', '192.168.122.163', '103.21.59.9'),
            conn(1446728712.45098, 'CCJwMV28pNkahEhhTg', '192.168.122.163', '103.21.59.9')
           ],
      L = 3,
      true.

%%%%%%%%%  Indicate end of test %%%%%%%%%%

test(final_db_proving_test, []) :-
      format('~nFinal db_proving Test ran~n').

:- end_tests(db_proving).

% end of db_proving.plt
