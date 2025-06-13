/*  db_extent.pl

(C) Amplify Intelligence, 2017

@copyright Steve Moyle

Test the extent of a Theory through a database.


|: db_extent.
This is a stub for db_extent

    DB Extent Options:
        -> Available database to test the extent:
        -> [[1,pibro20160315],[2,cw4,default]]
        -> Enter the [Number]. of the database to test extent
        -> 'none'. to abort.
|: 1.
Handle database [1]

    Testing Hypothesis

    [hypothesis]
malware_fetch(A,B,C) :-
   http_host(A,B,C), http_response_body_len(A,B,D), gt1000(D).


% asserting malware_fetch(_1436,_1438,_1440) :- http_host(_1436,_1438,_1440),http_response_body_len(_1436,_1438,_1468),gt1000(_1468) at reference <clause>(0x2cfe790).
% Erased clause at reference <clause>(0x2cfe790).
    Against database [1,pibro20160315]
     3940 goals provable/covered by hypothesis.

    Options:
        -> Enter the [Number] of goals to display (default 10)
        -> 'none'. to abort.
|: 20.
      Number of goals to display [20]
         20 random goals [[2109-malware_fetch(1458027762.12799,'Cb93wZeU1UiVRPtma','gb.archive.ubuntu.com'),2632-malware_fetch(1458059213.55801,'CpouAz3PrwWKLp2iac','gb.archive.ubuntu.com'),2305-malware_fetch(1458031322.10795,'C26nJbRZHu7lTC7U5','gb.archive.ubuntu.com'),3382-malware_fetch(1458064997.25801,'C0eckz9AmjKjZtRZ6','ichef.bbci.co.uk'),1899-malware_fetch(1458079645.62793,'CNjWnQ2rOjvhAL69sj','static.bbci.co.uk'),562-malware_fetch(1458055950.17799,'Cwiszm2TZprCI5rJOh','fonts.googleapis.com'),430-malware_fetch(1458055917.20795,'CM5our2PHAp0qAELGl','www.technewscentral.co.uk'),2912-malware_fetch(1458061888.02795,'Cnw7fhmRtg8p64ID5','static.boredpanda.com'),2885-malware_fetch(1458061883.98795,'Cbm7cg37fZYx8jFJ0i','static.boredpanda.com'),3593-malware_fetch(1458066136.29792,'Cx64W12c1E7YH2sPM2','static.pubmed.gov'),422-malware_fetch(1458054544.43807,'CvLGet1alRQnngcOD2','radio.thelounge.com'),3392-malware_fetch(1458064997.858,'Chxthv3z38YHWfQZMk','ichef.bbci.co.uk'),483-malware_fetch(1458055918.97796,'CEgwBC3IU1tGB6huXb','www.technewscentral.co.uk'),2496-malware_fetch(1458055929.56798,'CSkFA2LHrJ3uB3ZA6','a.disquscdn.com'),877-malware_fetch(1458061880.60794,'Cbm7cg37fZYx8jFJ0i','static.boredpanda.com'),1868-malware_fetch(1458075790.31796,'COxvN12uVwyQ135D71','static.bbci.co.uk'),2825-malware_fetch(1458061878.93797,'Cnw7fhmRtg8p64ID5','static.boredpanda.com'),1290-malware_fetch(1458064978.68796,'CR4sTF2cAYsKUtFgfk','static.bbc.co.uk'),3297-malware_fetch(1458064980.39793,'CPZy294dQT7l0Q2Ei5','www.bbc.co.uk'),1048-malware_fetch(1458062037.59793,'CiaGEg2efFeSGl0IA4','platform.twitter.com')]]

[done]


*/

:- module(db_extent, [
              db_extent/0

          ]).

:- use_module('../acuity_utils').

:- use_module(db_proving).
:- use_module(bottom_model).

:- use_module(library(lists)).

:- use_module(library(debug)).

% :- debug(assert_hyp).


db_extent :-
        db_extent_selection_menu(Menu),
	format('~n~t~4|DB Extent Options:~n'),
        format('~t~8|-> Available databases to test extent from:~n'),
        format('~t~8|-> ~w~n', [Menu]),
	format('~t~8|-> Enter the [Number]. of the database to test extent~n'),
	format('~t~8|-> \'none\'. to abort.~n'),
	repeat,
	read(DBNum),
	( DBNum = none
	->  format('~t~8| Aborting operation~n')
	;
	    member(DBAtom, Menu),
            [DBNum | _] = DBAtom,
	    format('~n~t~8|Using database [~w]~n', [DBNum]),
	    ask_db_extent(DBAtom)
	;
	    format('~t~8|Error: No database with index [~q]~n', [DBNum]),
	    format('~t~8|-> Enter the [Number] of the database to test extent~n'),
	    fail
	),
	!,
	true.

/*
ask_db_extent([1,pibro20160315]).
*/
ask_db_extent(DBAtom):-
	format('~n~t~4|Testing Hypothesis ~n~n~t~4|', []),
        show(hypothesis),
	format('~n~t~4|Against database ~t~4|~q', [DBAtom]),
        default_db(DefaultDB0),   % Setup the database to prove through
        (   DBAtom = [_0Num, DBName, default]
        ->  true % no need to switch DBs
        ;   DBAtom = [_0Num, DBName],
            register_db_for_proving(DBName)
        ),
        format('~n~t~4|(This may take some time.) ~t~4|', []),
        hypothesis(Head, Body, _),   % Setup the hypothesis to prove through
        db_extent_current_hypothesis((Head :- Body), Goals, Count),
	format('~n~t~4| ~w goals provable/covered by hypothesis.~n', [Count]),
	format('~n~t~4|Options:~n'),
	format('~t~8|-> Enter the [Number] of goals to display (default 10)~n'),
	format('~t~8|-> \'none\'. to abort.~n'),
	repeat,
	read(ItemsN),
	( ItemsN = none
	->  format('~t~8| Aborting operation~n')
	;   format('~t~8| Number of goals to display [~w]~n', [ItemsN]),
            (   between(1, Count, ItemsN) -> ItemsN = Items ; Items is 10), % Default hard coded here
	    list_with_index(Goals, GoalsWithIndex),
            random_nitems_from_indexed_list(Items, GoalsWithIndex, NRandomGoals),
	    format('~t~8| ~w random goals [~q]~n', [Items, NRandomGoals]),
            ask_import_base_via_saturation(NRandomGoals, DBName)
	;   false
	),
	!,
        default_db(DefaultDB1),   % Set the database for proving back to the initial database
        (   DefaultDB1 = DefaultDB0
        ->  true
        ;   register_db_for_proving(DefaultDB0)
        ),
	true.


/*
?- '$acuity_global'(db_for_proving, D).
D = cw4.

*/
default_db(DefaultDB) :-
    '$acuity_global'(db_for_proving, DefaultDB).

/*
?- findall(DB, odbc_current_connection(DB, _DSN), RDBs).
RDBs = [pibro20160315, cw4].
*/
registered_dbs(RDBs) :-
    findall(DB, odbc_current_connection(DB, _DSN), RDBs),
    true.

/*
?- db_extent_selection_menu(Menu).
Correct to: "db_extent:db_extent_selection_menu(Menu)"? yes
Menu = [[1, pibro20160315], [2, cw4, default]].

*/


db_extent_selection_menu(Menu) :-
    registered_dbs(RDBs),
    (   RDBs = []
    -> format('Warning: No registered databases for extent', [])
    ;   true
    ),
    (   default_db(DefaultDB)
    ->   true
    ;   format('Warning: No default database for extent', []),
        fail
    ),
    selection_menu(RDBs, DefaultDB, 0, Menu),
    true.



selection_menu([], _DefaultDB,  _N, []).

selection_menu([RDB | RDBs], DefaultDB, N, [MenuItem | Menu]) :-
    N1 is N + 1,
    (   RDB \= DefaultDB
    ->  MenuItem = [N1, RDB]
    ;   MenuItem = [N1, RDB, default]
    ),
    selection_menu(RDBs, DefaultDB, N1, Menu),
    true.




/*
db_extent_current_hypothesis(-[Goals], -Count)
*/

db_extent_current_hypothesis((Head :- Body), Goals, Count) :-
    assert_current_hypothesis(Head,  Body, Ref),
    findall(Head, call(Head), Goals),
    length(Goals, Count),
    !,
    erase(Ref),
    debug(assert_hyp, 'Erased clause at reference ~q.', [Ref]),
    true.

/*

?- show(hypothesis).
[hypothesis]
malware_fetch(A,B,C) :-
   http_host(A,B,C).
true.

?- hypothesis(H, A, B).
H = malware_fetch(_956, _958, _960),
A = http_host(_956, _958, _960).

*/
assert_current_hypothesis(Head, Body, Ref) :-
    asserta(user:(Head :- Body), Ref),
    debug(assert_hyp, 'asserting ~q :- ~q at reference ~q.', [Head, Body, Ref]),
    true.




ask_import_base_via_saturation(NRandomGoals, Database) :-
	format('~t~8|-> Enter the [Number] of the atom to saturate.~n'),
	format('~t~8|-> \'none\'. to abort.~n'),
        repeat,
        read(N),
        (   memberchk(N-Atom, NRandomGoals)
        ->  format('~t~8|Choice [~w]~n', [N]),
            format('~t~8|About to add atoms for ~q ~n.', [Atom]),
            format('~t~8|\'ok\'. to confirm~n'),
            format('~t~8|or \'none\'. to abort. ~n'),

            read(Ans),
            (   Ans = ok
            ->  Type = neg, %%%% MAY wish to make this a user choice between [pos, neg, uspec, ...]
                 import_base_via_saturation(Database, Atom, Type)
            ;    true
            )
        ;   N = none,
            format('Aborting')
        ;   format('Number ~w not in list, please chose a number.', [N]),
            fail
        ),
        true.


import_base_via_saturation(Database, Atom, Type) :-
        default_db(DefaultDB0),   % Setup the database to prove through
        (   Database = DefaultDB0
        ->  true % no need to switch DBs
        ;   register_db_for_proving(Database)
        ),
        import_base_via_saturation(Atom, Type),
        default_db(DefaultDB1),   % Set the database for proving back to the initial database
        (   DefaultDB1 = DefaultDB0
        ->  true
        ;   register_db_for_proving(DefaultDB0)
        ),
	true.



/*  db_extent.pl */


