/* eff_trans.pl
Steve Moyle, Amplify Intelligence,
    (C) 2017

TO-Do
-- Identify some utility predicates for moving to a library


NOTES
=====

This prototypes the rewriting of Hypthesis clauses so they can be
execuited more efficiencly.

1. Finding subsets of literals that are independent w.r.t. variabiles,
and can therefore be executeed independely (i.e. in parallel).

2. A start at combining binary/tertiary RDBMS predicates into a single
projection so that it requires fewer calls to the RDBMS.

Instead of doing

select ts, uid, host from http; followed by
select ts, uid, uri from http

replace with the obvious
select ts, uid, host, uri from http

The prototype (below) builds up structure that could easily be passed
through the CQL library.

e.g. the literals

http_host(F,G,A),http_uri(F,G,H)
get mapped to:
cql_input(http-[ts-F, uid-G, host-A, uri-H]),

which might then be transformed to (not implemented yet):

http_ts_uid_host_uri(F, G, A, H) :-
    {[], http [ts-F, uid-G, host-A, uri-H]}.


The body of the codde is from:

From: International Conference on Inductive Logic Programming
ILP 2000: Inductive Logic Programming pp 225-242
A Note on Two Simple Transformations for Improving the Efficiency of an ILP System
Vítor Santos Costa Ashwin Srinivasan Rui Camacho



A couple of examples -- orinal Costa et al at work.

?- reorder_clause((a(A) :- b(A,B), c(B,C), d(A,F), e(E,F)), Z).
Z =  (a(A):-(b(A, B), c(B, C)), !, d(A, F), e(E, F)).

Note the additional cut indicating the independence discovered.

calculate_independent_sets(D1, [], E1)
D1=
[ g(1,[A],b(C,A)),
  g(2,[A,B],c(A,B)),
  g(3,[D],d(C,D)),
  g(4,[D,E],e(E,D))
]

E1=
[ [ [A,B],
    g(1,[A],b(D,A)),
    g(2,[A,B],c(A,B))
  ],
  [ [C,E],
    g(3,[C],d(D,C)),
    g(4,[C,E],e(E,C))
  ]
]


Something from the Malware domain example.

?- reorder_clause((malware_host(A) :-    http_host(B,C,A), http_uri(B,C,D), uri_param(D,E), http_host(F,G,A), http_uri(F,G,H), uri_param(H,I), I\=E), Z).
Z =  (malware_host(A):-http_host(B, C, A), http_host(F, G, A), http_uri(F, G, H), uri_param(H, I), http_uri(B, C, D), uri_param(D, E), I\=E).

calculate_independent_sets(D1, [], E1)
D1=
[ g(1,[A,B],http_host(A,B,G)),
  g(2,[A,B,C],http_uri(A,B,C)),
  g(3,[C,D],uri_param(C,D)),
  g(4,[E,F],http_host(E,F,G)),
  g(5,[E,F,H],http_uri(E,F,H)),
  g(6,[H,I],uri_param(H,I)),
  g(7,[D,I],I\=D)
]

E1=
[ [ [A,B,H,I,C,D,F,G],
    g(1,[A,B],http_host(A,B,E)),
    g(4,[C,D],http_host(C,D,E)),
    g(5,[C,D,F],http_uri(C,D,F)),
    g(6,[F,G],uri_param(F,G)),
    g(2,[A,B,H],http_uri(A,B,H)),
    g(3,[H,I],uri_param(H,I)),
    g(7,[I,G],G\=I)
  ]
]



THIS IS WHERE I GOT TO:
?- join_binary_predicates((malware_host(A) :-    http_host(B,C,A), http_uri(B,C,D), uri_param(D,E), http_host(F,G,A), http_uri(F,G,H), uri_param(H,I), I\=E), Z), write_term(Z, [numbervars(false)]).
Correct to: "eff_trans:join_binary_predicates((malware_host(A):-http_host(B,C,A),http_uri(B,C,D),uri_param(D,E),http_host(F,G,A),http_uri(F,G,H),uri_param(H,I),I\\=E),Z)"? yes
% http-[ts-_92918,uid-_92920,host-_92914]
% http-[ts-_92918,uid-_92920,uri-_92930]
% compat_goals: [ts-_92918,uid-_92920,host-_92914], [ts-_92918,uid-_92920,uri-_92930]
% Compat: g(1,[_92918,_92920],http_host(_92918,_92920,_92914)) : [g(2,[_95568,_95574,_95580],http_uri(_95568,_95574,_95580))]
% http-[ts-_92918,uid-_92920,host-_92914]
% http-[ts-_95568,uid-_95574,uri-_95580]
% http-[ts-_92940,uid-_92942,host-_92914]
% http-[ts-_92940,uid-_92942,uri-_92952]
% compat_goals: [ts-_92940,uid-_92942,host-_92914], [ts-_92940,uid-_92942,uri-_92952]
% Compat: g(4,[_92940,_92942],http_host(_92940,_92942,_92914)) : [g(5,[_96148,_96154,_96160],http_uri(_96148,_96154,_96160))]
% http-[ts-_92940,uid-_92942,host-_92914]
% http-[ts-_96148,uid-_96154,uri-_96160]
% Compat: g(6,[_92952,_92958],uri_param(_92952,_92958)) : []
% Compat: g(3,[_92930,_92936],uri_param(_92930,_92936)) : []
% Compat: g(7,[_92936,_92958],_92958\=_92936) : []
malware_host(_92914):-_92958\=_92936,uri_param(_92930,_92936),uri_param(_92952,_92958),cql_input(http-[ts-_92940,uid-_92942,host-_92914,uri-_92952]),cql_input(http-[ts-_92918,uid-_92920,host-_92914,uri-_92930])
Z =  (malware_host('$VAR'('A')):-'$VAR'('I')\='$VAR'('E'), uri_param('$VAR'('D'), '$VAR'('E')), uri_param('$VAR'('H'), '$VAR'('I')), cql_input(http-[ts-'$VAR'('F'), uid-'$VAR'('G'), host-'$VAR'('A'), uri-'$VAR'('H')]), cql_input(http-[ts-'$VAR'('B'), uid-'$VAR'('C'), host-'$VAR'('A'), uri-'$VAR'('D')])).


IE:


Z =  (
malware_host(A):-
I\=E, uri_param(D, E), uri_param(H, I),
cql_input(http-[ts-F, uid-G, host-A, uri-H]),
cql_input(http-[ts-B, uid-C, host-A, uri-D])
).


*/

:- module(eff_trans, [
              remove_redundant/2,  % (Head:-Body), (Head1:-Body1)
              reorder_clause/2,     % (Head:-Body), Clause
              join_binary_predicates/2 %((Head:-Body), (Head:-NewBody))
          ]).


:- use_module(library(lists)).
:- use_module(library(ordsets)).
:- use_module(library(pairs)).


:- use_module(library(debug)).
% :- debug(independent_goals).
% :- debug(compat_goals).
% :- debug(db_pred).

table_name('http').   % FOR EXAMPLE

join_binary_predicates((Head:-Body), (Head:-NewBody)) :-
    term_variables(Head,LHead),
    number_goals_and_get_vars(Body,LHead,1,_,[],Conj),
    calculate_independent_sets(Conj,[],BSets),
    compile_independents(BSets, Gs),
    flatten(Gs, NewBodyList),
    list_to_goals(NewBodyList, NewBody),
    true.


/*
[ [ [A,B,H,I,C,D,F,G],
    g(1,[A,B],http_host(A,B,E)),
    g(4,[C,D],http_host(C,D,E)),
    g(5,[C,D,F],http_uri(C,D,F)),
    g(6,[F,G],uri_param(F,G)),
    g(2,[A,B,H],http_uri(A,B,H)),
    g(3,[H,I],uri_param(H,I)),
    g(7,[I,G],G\=I)
  ]
]
*/
compile_independents([], []).
compile_independents([Bset | BSets], [Goals | Gs]):-
    compile_independent(Bset, Goals),
    compile_independents(BSets, Gs).

compile_independent(BSet, Goals) :-
    BSet = [_Vars | GoalTerms],
    independent_goals(GoalTerms, [], Goals),
    true.

/*
?- independent_goals([ g(1,[A,B],http_host(A,B,E)), g(4,[C,D],http_host(C,D,E)), g(5,[C,D,F],http_uri(C,D,F)), g(6,[F,G],uri_param(F,G)), g(2,[A,B,H],http_uri(A,B,H)), g(3,[H,I],uri_param(H,I)), g(7,[I,G],G\=I)], [], Z), write_canonical(Z).
Correct to: "eff_trans:independent_goals([g(1,[A,B],http_host(A,B,E)),g(4,[C,D],http_host(C,D,E)),g(5,[C,D,F],http_uri(C,D,F)),g(6,[F,G],uri_param(F,G)),g(2,[A,B,H],http_uri(A,B,H)),g(3,[H,I],uri_param(H,I)),g(7,[I,G],G\\=I)],[],Z)"? yes
% http-[ts-_55418,uid-_55424,host-_55434]
% http-[ts-_55418,uid-_55424,uri-_55570]
% compat_goals: [ts-_55418,uid-_55424,host-_55434], [ts-_55418,uid-_55424,uri-_55570]
% Compat: g(1,[_55418,_55424],http_host(_55418,_55424,_55434)) : [g(2,[_57524,_57530,_57536],http_uri(_57524,_57530,_57536))]
% http-[ts-_55418,uid-_55424,host-_55434]
% http-[ts-_57524,uid-_57530,uri-_57536]
% http-[ts-_55452,uid-_55458,host-_55434]
% http-[ts-_55452,uid-_55458,uri-_55498]
% compat_goals: [ts-_55452,uid-_55458,host-_55434], [ts-_55452,uid-_55458,uri-_55498]
% Compat: g(4,[_55452,_55458],http_host(_55452,_55458,_55434)) : [g(5,[_58104,_58110,_58116],http_uri(_58104,_58110,_58116))]
% http-[ts-_55452,uid-_55458,host-_55434]
% http-[ts-_58104,uid-_58110,uri-_58116]
% Compat: g(6,[_55498,_55532],uri_param(_55498,_55532)) : []
% Compat: g(3,[_55570,_55604],uri_param(_55570,_55604)) : []
% Compat: g(7,[_55604,_55532],_55532\=_55604) : []
[\=(B,A),uri_param(E,A),uri_param(C,B),cql_input(-(http,[-(ts,_),-(uid,_),-(host,D),-(uri,C)])),cql_input(-(http,[-(ts,_),-(uid,_),-(host,D),-(uri,E)]))]

*/
independent_goals([], Gs, Gs).
independent_goals([GT1 | GoalTerms], Goals0, Goals) :-
    findall(GT, (member(GT, GoalTerms),
                 compat_goals(GT1, GT)
                ), GTs),
    debug(independent_goals, 'Compat: ~q : ~q', [GT1, GTs]),
    Compatible = [GT1 | GTs],
    build_projection(Compatible, CompGoals),
    subtract(GoalTerms, GTs, GoalTerms1),
    independent_goals(GoalTerms1, [CompGoals | Goals0], Goals).


% build_projection(L, L).  % Temporary defn

build_projection([G1| Gs], Proj) :-
     G1 = g(_Id1, _Vars1, Lit1),
     (   db_pred(Lit1, Table, CQLPairs)
     ->   build_projection_1(Gs, Table, CQLPairs, CQLPairs1),
          deduplicate(CQLPairs1, CQLPairsShort),
          Proj = cql_input(Table - CQLPairsShort)
     ;    Proj = Lit1
     ).

build_projection_1([], _Table, CQLPairs, CQLPairs) :-
   true.
build_projection_1([G | Gs], Table, Pairs0, Cols0) :-
    G = g(_Id1, _Vars1, Lit1),
    db_pred(Lit1, Table, CQLPairs),
    append(Pairs0, CQLPairs, Pairs1),
    build_projection_1(Gs, Table, Pairs1, Cols0).


/*
?- compat_goals(g(4,[C,D],http_host(C,D,E)), g(5,[C,D,F],http_uri(C,D,F))).
Correct to: "eff_trans:compat_goals(g(4,[C,D],http_host(C,D,E)),g(5,[C,D,F],http_uri(C,D,F)))"? yes
true.

?- compat_goals(g(4,[C,D],http_host(C,D,E)), g(2,[A,B,H],http_uri(A,B,H))).
Correct to: "eff_trans:compat_goals(g(4,[C,D],http_host(C,D,E)),g(2,[A,B,H],http_uri(A,B,H)))"? yes
false.
*/
compat_goals(G1, G2) :-
    G1 = g(_Id1, Vars1, Lit1),
    G2 = g(_Id2, Vars2, Lit2),
    ord_subset(Vars1, Vars2), % Vars match
    db_pred(Lit1, Table, CQLPairs1),
    db_pred(Lit2, Table, CQLPairs2),
    debug(compat_goals,'compat_goals: ~q, ~q', [CQLPairs1, CQLPairs2]),
    true.


/*
?- db_pred(http_uri(A, B, C), T, Cs).
T = http,
Cs = [ts, uid, uri].
*/
db_pred(Lit, Table, CQLPairs) :-
    Lit =..[Pred | Vars],
    atomic_list_concat([Table, Col], '_', Pred),
    table_name(Table),
    Cols = [ts, uid, Col],
    pairs_keys_values(CQLPairs,Cols, Vars),
    debug(db_pred, '~w-~w', [Table, CQLPairs]),
    true.



% Transformation 1
remove_redundant((Head:-Body),(Head1:-Body1)):-
    goals_to_list((Head,Body),ClauseL),
    remove_subsumed(ClauseL,[Head1|Body1L]),
    (Body1L = []
    -> Body1 = true
    ;
       list_to_goals(Body1L,Body1)
    ).

% Transformation 2
reorder_clause((Head:-Body), Clause) :-
    term_variables(Head,LHead),
    number_goals_and_get_vars(Body,LHead,1,_,[],Conj),
    calculate_independent_sets(Conj,[],BSets),
    compile_clause(BSets,Head,Clause).

number_goals_and_get_vars((G,Body),LHead,I0,IF,L0,[g(I0,LVF,NG)|LGs]) :- !,
    I is I0+1,
    get_goal_vars(G,LHead,LVF,NG),
    number_goals_and_get_vars(Body,LHead,I,IF,L0,LGs).

number_goals_and_get_vars(G,LHead,I,I,L0,[g(I,LVF,NG)|L0]) :-
    get_goal_vars(G,LHead,LVF,NG).

get_goal_vars(G,LHead,LVF,G) :-
    term_variables(G,LV0),
    sort(LV0,LVI),
    ord_subtract(LVI,LHead,LVF).

calculate_independent_sets([],BSets,BSets).
calculate_independent_sets([G|Ls],BSets0,BSetsF) :-
    add_goal_to_set(G,BSets0,BSetsI),
    calculate_independent_sets(Ls,BSetsI,BSetsF).

add_goal_to_set(g(I,LV,G),Sets0,SetsF) :-
    add_to_sets(Sets0,LV,[g(I,LV,G)],SetsF).

add_to_sets([],LV,Gs,[[LV|Gs]]).
add_to_sets([[LV|Gs]|Sets0],LVC,GsC,[[LV|Gs]|SetsF]) :-
    ord_disjoint(LV,LVC), !,
    add_to_sets(Sets0,LVC,GsC,SetsF).
add_to_sets([[LV|Gs]|Sets0],LVC,GsC,SetsF) :-
    ord_union(LV,LVC,LVN),
    join_goals(Gs,GsC,GsN),
    add_to_sets(Sets0,LVN,GsN,SetsF).

join_goals([],L,L):- !.
join_goals(L,[],L):- !.
join_goals([g(I1,VL1,G1)|T],[g(I2,VL2,G2)|T2],Z) :-
    I1 < I2, !,
    Z = [g(I1,VL1,G1)|TN],
    join_goals(T,[g(I2,VL2,G2)|T2],TN).
join_goals([H|T],[g(I2,VL2,G2)|T2],Z) :-
    Z = [g(I2,VL2,G2)|TN],
join_goals(T,[H|T2],TN).

compile_clause(Goals,Head,(Head:-Body)):-
    compile_clause2(Goals,Body).

compile_clause2([[_|B]], B1):-
    !,
    glist_to_goals(B,B1).
compile_clause2([[_|B]|Bs],(B1,!,NB)):- % Note addition of the cut in the output
    glist_to_goals(B,B1),
    compile_clause2(Bs,NB).

% remove literals subsumed in the body of a clause
remove_subsumed([Head|Lits],Lits1):-
    delete(Lit,Lits,Left),
    \+(\+(redundant(Lit,[Head|Lits],[Head|Left]))), !,
    remove_subsumed([Head|Left],Lits1).
remove_subsumed(L,L).

% determine if Lit is subsumed by a body literal
redundant(Lit,Lits,[Head|Body]):-
    copy_term([Head|Body],Rest1),
    member(Lit1,Body),
    Lit = Lit1,
    subsumes(Lits,Rest1).

subsumes(Lits,Lits1):-
    \+(\+((numbervars(Lits,0,_),numbervars(Lits1,0,_),subset1(Lits,Lits1)))).


% General utilities
list_to_goals([Goal],Goal):- !.
list_to_goals([Goal|Goals],(Goal,Goals1)):-
    list_to_goals(Goals,Goals1).

glist_to_goals([g(_,_,Goal)],Goal):- !.
glist_to_goals([g(_,_,Goal)|Goals],(Goal,Goals1)):-
    glist_to_goals(Goals,Goals1).
goals_to_list((true,Goals),T):-
    !,
    goals_to_list(Goals,T).
goals_to_list((Goal,Goals),[Goal|T]):-
    !,
    goals_to_list(Goals,T).
goals_to_list(true,[]):- !.
goals_to_list(Goal,[Goal]).

subset1([],_).
subset1([Elem|Elems],S):-
    member1(Elem,S), !,
    subset1(Elems,S).

member1(H,[H|_]):- !.
member1(H,[_|T]):-
    member1(H,T).


/* member(H,[H|_]).
member(H,[_|T]):-
    member(H,T).

delete(H,[H|T],T).
delete(H,[H1|T],[H1|T1]):-
    delete(H,T,T1).
*/


deduplicate([], []).
deduplicate([L | Long], [L | Short]) :-
    delete(Long, L, NoLinLong),
    deduplicate(NoLinLong, Short).

/* end of eff_trans.pl */


