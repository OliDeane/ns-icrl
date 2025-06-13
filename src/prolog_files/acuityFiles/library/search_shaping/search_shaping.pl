/* search_shaping.pl

    Author:        Steve Moyle
    Company:       Amplify Intelligence
    E-mail:        katarapkokid@hotmail.com
    Copyright (c)  2017, 2022
    All rights reserved.


Library to interactively develop controls on the Aleph search space.

E.g. with prunes and false. Show it used on the command line. 

?- search_shaping((malware_host(A) :-http_host(B,C,A), http_uri(B,C,D), uri_file_name(D,E), uri_param(D,F), http_method(B,C,'POST'))).

    Hypothesis to choose elements from:

    malware_host(A) :-
    [1]     http_host(B,C,A),
    [2]     http_uri(B,C,D),
    [3]     uri_file_name(D,E),
    [4]     uri_param(D,F),
    [5]     http_method(B,C,'POST').


    Search Shaping Options:

     - Hypotheses Must contain literals 'must([i, .., n]).'
     - Hypotheses Must NOT contain literals 'must_not([j, .., m]).'
        -> Enter the must or must_not list.
        -> 'none'. to abort.
|: must([2,4,5]).

        Shaping Term is sensible [must([2,4,5])]

-8|Proposed constraint is:



-8|     constraint_clause(1, true,[http_uri/3,uri_param/2,http_method/3],must,[0-malware_host(A),1-http_host(B,C,A),2-http_uri(B,C,D),3-uri_file_name(D,E),4-uri_param(D,F),5-http_method(B,C,'POST')],[2,4,5])

        -> 'ok.' to accept constraint or 'none.' to abort.
|: ok.

% Constraint added at ref:  <clause>(0xff04c0).
A = A,
B = B,
C = C,
D = D,
E = E,
F = F.

?- listing(user:false).                                                         :- dynamic search_shaping:false/0.

%   NOTE: system definition has been overruled for false/0
% the acuity_global here is the way you stash loads of clauses and enable you to retract all of them. 
% Check all the constraints stored in acuity_global - and if any trigger false, then ...?
% You have this constraint and aleph will be testing it every time. 
:- dynamic false/0.

false :-
    search_shaping:
    (   user:hypothesis(_, A, _),
	goals_to_list(A, D),
	user:'$acuity_global'(constraint_spec, constraint_clause(_, true, B, C, _, _)),
	target_template_list(B, E),
	search_constraint(C, D, E),
	true
    ).

true.

?- listing('$acuity_global').
:- dynamic'$acuity_global'/2.

'$acuity_global'(constraint_spec, constraint_clause(1, true, [http_uri/3, uri_param/2, http_method/3], must, [0-malware_host(A), 1-http_host(B, C, A), 2-http_uri(B, C, D), 3-uri_file_name(D, E), 4-uri_param(D, F), 5-http_method(B, C, 'POST')], [2, 4, 5])).

true.


Notes:

    user:'$acuity_global'(constraint_spec,
                   constraint_clause(
                   ID, % Unique spec ID - allows you to fish individual constraints and turn on/off. 
                   [true | false], %  true => constraint activated ;
                   false => Constraint ignored during search
                   TargetPredAritys, %  List of pred/arity for considered clause
                   %  see example above 0 = Head
                   InOrOut, %  must | must_not
                   _OrigIDs)),



*/

:- module( search_shaping, [
               search_shaping/2,   % +(Head:-Body)
               delete_acuity_constraints/0
           ]).


:- use_module('../acuity_utils').

:- use_module(library(lists)).

:- use_module(library(debug)).

:- nodebug(constraint_spec).
:- nodebug(must).
:- nodebug(must_not).
:- nodebug(ground_must_match).
:- nodebug(unbound_constraint_spec).

:- redefine_system_predicate(user:false),
    dynamic(user:false/0).


user:false :-
    user:hypothesis(Head, Body, _),
    goals_to_list(Body, BodyList),
    goals_to_list(Head, HeadList),
    !,
    user:'$acuity_global'(constraint_spec,
                          constraint_clause(_CID, true, TargetPredAritys, MustOrNot, _RefClause, _OrigIDs)
                         ),
    target_template_list(TargetPredAritys, TargetList),
    extract_first_element(TargetList, TestTarget),
    debug(unbound_constraint_spec, 'Unbound: ~q ~w be in ~q', [TargetList, MustOrNot, BodyList]),
    search_constraint(MustOrNot, BodyList, TestTarget),
    debug(unbound_constraint_spec, ' -- Bound: ~q ~w be in ~q', [TargetList, MustOrNot, BodyList]),
    user:asserta('$pycuity_constraint_violations'((Head:-Body),MustOrNot,TargetList)),
   !,
   true.

mytest(MustOrNot, BodyList, TargetList) :-
    debug(unbound_constraint_spec, 'Unbound: ~q ~w be in ~q', [TargetList, MustOrNot, BodyList]),
    search_constraint(MustOrNot, BodyList, TargetList),
    debug(unbound_constraint_spec, ' -- Bound: ~q ~w be in ~q', [TargetList, MustOrNot, BodyList]),
   !,
   print(BodyList), print(TargetList),
   true.

delete_acuity_constraints :-
    retractall(user:'$acuity_global'(constraint_spec, _ConstraintClause)).

search_shaping(Clause,AcuityConstraintSpec) :-
    format('~n~t~4|Hypothesis to choose elements from:~n'),
    clause_with_index(Clause, ClauseListWithIndex),
    print_clause_for_shaping(ClauseListWithIndex),
    format('~n~t~4|Search Shaping Options:~n'),
    format('~n~t~4| - Hypotheses Must contain literals \'must([i, .., n]).\'', []), 
    format('~n~t~4| - Hypotheses Must NOT contain literals \'must_not([j, .., m]).\'', []),
    format('~n~t~8|-> Enter the must or must_not list. (e.g. |: must([1,1,5]). )'),
    format('~n~t~8|-> \'none\'. to abort.~n'),
    repeat,
    read(STerm), 
    ( STerm = none
    ->  format('~t~8| Aborting operation~n')
    ;
      check(STerm, ClauseListWithIndex),
      format('~n~t~8|Shaping Term is sensible [~w]~n', [STerm]),
      make_constraint_spec(STerm, ClauseListWithIndex, AcuityConstraintSpec),
      format('~n~t~8|Proposed constraint is: ~n', []),
      format('~n~t~8|     ~q~n', [AcuityConstraintSpec]),
      format('~n~t~8|-> \'ok.\' to accept constraint or \'none.\' to abort.~n'),
      read(Action),
      (   Action = ok
      ->  assert(user:'$acuity_global'(constraint_spec, AcuityConstraintSpec), DBRef),
          debug(constraint_spec, 'Constraint added at ref:  ~q.', [DBRef]),
          true
      ;   Action = none
      ;   fail
      )
    ;
      format('~t~8|Error: Invalid constraint specification.~n', []),
      format('~t~8|-> Enter the must or must_not list.~n'),
      fail
    ),
    !,
    true.



/*
constraint_clause gives some traceability as to which clause was used to
define the constraint.



constraint_clause(
    true,
    [uri_file_name/2, http_method/3],
    must,
    [0-malware_host(A),1-http_host(B,C,A),2-http_uri(B,C,D),3-uri_file_name(D,E),4-uri_param(D,F),5-http_method(B,C,'POST')],
    [3,5]
).

*/


/* Old version P/A only constraints (unbound)

make_constraint_spec(STerm, ClauseListWithIndex, AcuityConstraintSpec) :-
    STerm =..[MustOrNot, IDs],
    findall(Pred/Arity, (
                     member(ID, IDs),
                     member(ID-Functor, ClauseListWithIndex),
                     functor(Functor, Pred, Arity),
                true ),
            PASpecs),
    next_constraint_id(CID),
    AcuityConstraintSpec = constraint_clause(CID, true, PASpecs, MustOrNot, ClauseListWithIndex, IDs), !, true.
*/


make_constraint_spec(STerm, ClauseListWithIndex, AcuityConstraintSpec) :-
    STerm =..[MustOrNot, IDs],
    findall(Literal, (
                     member(ID, IDs),
                     member(ID-Literal, ClauseListWithIndex),
                true ),
            LitsSpecs),
    next_constraint_id(CID),
    AcuityConstraintSpec = constraint_clause(CID, true, LitsSpecs, MustOrNot, ClauseListWithIndex, IDs),
    !,
    true.

search_constraint(must_not, BodyList, TargetList) :-
    subsetchk(TargetList, BodyList),
    debug(must_not, 'Constraint vialoted. Clause body ~q must_not contain ~q.', [BodyList, TargetList]),
    true.

% We need to find the old version of this search_constraint - but this does work, but not firmly tested.
%search_constraint(must, BodyList, TargetList) :- \+ must_subsetchk(TargetList, BodyList). 
search_constraint(must, BodyList, TargetList) :- \+ check_subset(TargetList, BodyList). 


copy_list(TargetList, NewList) :-
    findall(Element, member(Element, TargetList), NewList).

extract_first_element([FirstElement | _], [FirstElement]).

target_template_list([], []) :-
    true.
target_template_list([P/A | TargetPredArity], [Pred | TargetList]) :-
    !,
    functor(Pred, P, A),
    target_template_list(TargetPredArity, TargetList).
target_template_list([Lit | TargetPredArity], [Lit | TargetList]) :-
    % not(Lit = P/A)
    % functor(Pred, P, A),
    target_template_list(TargetPredArity, TargetList).



print_clause_for_shaping(ClauseList) :-
    numbervars(ClauseList, 0, _),
    ClauseList = [0-Head | BodyListWithIndex],
    format('~n~t~4|~q :-~n', [Head]),
    print_body_for_shaping(BodyListWithIndex),
    true.

print_body_for_shaping([N-Lit]) :-
    !,  % Last element
    format('~t~4|[~w]     ~q.~n~n', [N, Lit]),
    true.
print_body_for_shaping([N-Lit | Lits]) :-
    format('~t~4|[~w]     ~q,~n', [N, Lit]),
    print_body_for_shaping(Lits).


check(Term, ClauseListWithIndex) :-
    Term =.. [Pred, IDs],
    (   Pred = must ; Pred = must_not),
    forall(member(ID, IDs), (integer(ID), ID > 0)), % reject the head at index 0
    check_indexes_exist(IDs, ClauseListWithIndex).

check_indexes_exist([], _ClauseListWithIndex) :-
    !.
check_indexes_exist([ID | IDs], ClauseListWithIndex) :-
    memberchk(ID-_Lit, ClauseListWithIndex),
    check_indexes_exist(IDs, ClauseListWithIndex),
    true.


/* candidate predicate to be moved to acuity_utils.pl */
clause_with_index(Clause, ClauseListWithIndex) :-
    clause_to_list(Clause, [Head | BodyList]),
    list_with_index(BodyList, BodyListWithIndex),
    ClauseListWithIndex = [0-Head | BodyListWithIndex].

subsetchk([], []):-  % equal subset
    !.
subsetchk([], Large) :- % Proper subset
    Large \=[],
    !.
subsetchk([S | Small], Large) :-
    struct_select(S, Large, Large1),
    !,
    subsetchk(Small, Large1).


%!  struct_select(?Elem, ?List1, ?List2)
%
%   Is true when List1,  with  Elem   removed,  results  in  List2. This
%   implementation is determinsitic if the  last   element  of List1 has
%   been selected AND they are structurally equal (not simply
%   unificaiton equal).

struct_select(Elem, [Head|Long], Short) :-
    struct_select3_(Long, Head, Elem, Short).

struct_select3_(Short, Elem, Elem, Short).
struct_select3_([Head2|Long], Head, Elem, [Head|Short]) :-
    struct_select3_(Long, Head2, Elem, Short).



%!  must_subsetchk(+MustPreds, +BodyPreds)
%
%   Is true when all MustPreds appear in BodyPreds such that
%   ground elements of MustPreds literals already appear in
%   corresponding BodyPreds literals.

must_subsetchk([], []):-  % equal subset
    !.
must_subsetchk([], Large) :- % Proper subset
    Large \=[],
    !.
must_subsetchk([S | Small], Large) :-
    must_struct_select(S, Large, Large1),
    !,
    must_subsetchk(Small, Large1).

must_struct_select(Elem, [Head|Long], Short) :-
    must_struct_select3_(Long, Head, Elem, Short).

must_struct_select3_(Short, Elem1, Elem2, Short):-
    ground_must_match(Elem2, Elem1).

must_struct_select3_([Head2|Tail], Head, Elem, [Head|Rest]) :-
    must_struct_select3_(Tail, Head2, Elem, Rest).

%%% My code 
check_subset([], _).
check_subset([X|Xs], List) :-
    member(X, List),
    substitute_vars(Xs, List).

substitute_vars([], _).
substitute_vars([X|Xs], List) :-
    (var(X) -> member(Y, List), substitute_var(X, Y) ; X = Y),
    substitute_vars(Xs, List).

substitute_var(Var, Val) :-
    (var(Var) -> Var = Val ; Var == Val).

%%%

:- multifile user:show/1.

user:show(shaping) :-
    format('Hypothesis Shaping Constraints: ~n~n', []),
    format('- Must contain ~n', []),
    format('   [active Constraints]~n', []),
    show_active_shaping_musts,
    format('   [In-active Constraints]~n', []),
    show_inactive_shaping_musts,
    format('~n- Must NOT contain ~n', []),
    format('   [active Constraints]~n', []),
    show_active_shaping_must_nots,
    format('   [In-active Constraints]~n', []),
    show_inactive_shaping_must_nots,
    true.


show_shaping(Type, activeState) :-
    member(Type-MsgType, [must-'MUST', must_not-'MUST NOT']),
    member(activeState, [true, false]),
    user:'$acuity_global'(constraint_spec,
                          constraint_clause(CID,
                                            activeState,
                                            MustNameArityLits,
                                            Type,
                                            IndexedPickClause,
                                            _MustIndices)),
    member(0-HeadAtom, IndexedPickClause),
    functor(HeadAtom, HeadPred, HeadArity),
    HeadNameArity = HeadPred/HeadArity,
    format('ID:[~w] Clauses with head ~w ~w contain at least one of each of ~w in the body.~n',
           [CID, HeadNameArity, MsgType, MustNameArityLits]),
    true.

show_active_shaping_musts :-
    show_shaping(must, true),
    fail.
show_active_shaping_musts.
show_inactive_shaping_musts :-
    show_shaping(must, false),
    fail.
show_inactive_shaping_musts.


show_active_shaping_must_nots :-
    show_shaping(must_not, true),
    fail.
show_active_shaping_must_nots.
show_inactive_shaping_must_nots :-
    show_shaping(must_not, false),
    fail.
show_inactive_shaping_must_nots.



constraint_ids(IDs) :-
    findall(ID,
            user:'$acuity_global'(constraint_spec,
                          constraint_clause(ID,
                                            _TrueOrFalse,
                                            _MustNotNameArityLits,
                                            _Type,
                                            _IndexedPickClause,
                                            _SpecIndices)),
            IDs).

next_constraint_id(ID) :-
    constraint_ids(Ids),
    (   Ids = []   % No registered constraints
    ->  ID = 1
    ;
        sort(Ids, SIds),
        reverse(SIds, [PrevID|_]),
        ID is PrevID + 1
    ),
    assertion(\+ memberchk(ID, Ids)).



set_constraint_effect(CID, TrueOrFalse) :-
    ground(TrueOrFalse),
    memberchk(TrueOrFalse, [true, false]),
    constraint_ids(IDs),
    (   memberchk(CID, IDs)
    ->  retract(user:'$acuity_global'(constraint_spec,
                          constraint_clause(CID,
                                            _OldTrueOrFalse,
                                            MustNotNameArityLits,
                                            Type,
                                            IndexedPickClause,
                                            SpecIndices))),
         assert(user:'$acuity_global'(constraint_spec,
                          constraint_clause(CID,
                                            TrueOrFalse,
                                            MustNotNameArityLits,
                                            Type,
                                            IndexedPickClause,
                                            SpecIndices)))
    ;
        format('Warning: Constraint with index ~w does not exist~n', [])
    ).

delete_constraint(CID) :-
    constraint_ids(IDs),
    (   memberchk(CID, IDs)
    ->  retract(user:'$acuity_global'(constraint_spec,
                          constraint_clause(CID,
                                            _OldTrueOrFalse,
                                            _MustNotNameArityLits,
                                            _Type,
                                            _IndexedPickClause,
                                            _SpecIndices))),
        format('Constraint with index ~w deleted~n', [CID])
    ;
        format('Warning: Constraint with index ~w does not exist~n', [CID])
    ).




%!  ground_must_match(+MustTerm, +MatchTerm)
%!	is det.
%
%   The ground elements in the MustTerm must appear in the
%   MatchTerm also ground.  That is unification cannot be use
%   to bind a variable in the MatchTerm to the ground elements
%   in the MustTerm
%   examples
/*
%   ?- ground_must_match(f(A), f(B)).
A = B.

?- ground_must_match(f(A), f(dns)).
A = dns.

?- ground_must_match(f(dns), f(dns)).
true.

?- ground_must_match(f(dns), f(X)).
false.
*/

ground_must_match(MustTerm, MatchTerm) :-
    debug(ground_must_match, 'Testing: that must ~q  matches ~q', [MustTerm, MatchTerm]),
    args_match([MustTerm], [MatchTerm]).

args_match([], []).

% Case 1) If LHS is ground, then we may not let RHS unify unless it is
% also ground AND equivalent
args_match([MuA | MuAs], [MaA | MaAs]) :-
    ground(MuA),
    !,
    ground(MaA),
    MuA == MaA,
    args_match(MuAs, MaAs).

% Case 2) If LHS is a variable, then RHS may unify it
args_match([MuA | MuAs], [MaA | MaAs]) :-
    var(MuA),
    !,
    MuA = MaA,
    args_match(MuAs, MaAs).


% Case 3) If LHS is a term
args_match([MuA | MuAs], [MaA | MaAs]) :-
    MuA =.. [F | MuAVars],
    !,
    MaA =.. [F | MaAVars],
    args_match(MuAVars, MaAVars),
    args_match(MuAs, MaAs).


% Below is OD's fix for the must_not constraint.
search_constraint(must, BodyList, TargetList) :-
                \+ check_subset(TargetList, BodyList).


check_subset([], _).
check_subset([X|Xs], List) :-
    member(X, List),
    substitute_vars(Xs, List).

substitute_vars([], _).
substitute_vars([X|Xs], List) :-
    (var(X) -> member(Y, List), substitute_var(X, Y) ; X = Y),
    substitute_vars(Xs, List).

substitute_var(Var, Val) :-
    (var(Var) -> Var = Val ; Var == Val).





/* Uncomment for stand alone testing
%%%%%%%%% From aleph6.pl %%%%%
clause_to_list((Head:-true),[Head]):- !.
clause_to_list((Head:-Body),[Head|L]):-
        goals_to_list(Body,L).
clause_to_list(Head,[Head]).
list_to_clause([Goal],(Goal:-true)):- !.
list_to_clause([Head|Goals],(Head:-Body)):-
	list_to_goals(Goals,Body).
list_to_goals([Goal],Goal):- !.
list_to_goals([Goal|Goals],(Goal,Goals1)):-
	list_to_goals(Goals,Goals1).
goals_to_list((true,Goals),T):-
	goals_to_list(Goals,T).
goals_to_list((Goal,Goals),[Goal|T]):-
	goals_to_list(Goals,T).
goals_to_list(true,[]):- !.
goals_to_list(Goal,[Goal]).
*/
%%%%%%%%% From aleph6.pl %%%%%


/* end of search_shaping.pl  */


