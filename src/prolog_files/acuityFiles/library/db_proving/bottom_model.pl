/* bottom_model.pl

Produce a set of atoms that were proved (within the mode language)
during the creation of bottom.

?- '$aleph_sat_litinfo'(A, B, C, D, E, F).
A = 2,
B = 0,
C = has_car(1, 2),
D = [[1]/train],
E = [[2]/car],
F = [10, 13, 17, 21, 25] ;

?- '$aleph_sat_terms'(A, B, C, D).

A = 1,
B = 0,
C = east1,
D = train.

A = 2,
B = 1,
C = car_11,
D = car ;



A = 25,
B = 0,
C = load(2, aleph_const(rectangle), aleph_const(3)),
D = [[1]/car],
E = F, F = [].


A = 6,
B = 2,
C = rectangle,
D = shape ;

*/

:- module(bottom_model, [
              import_base_via_saturation/2  %+SeedAtom, +SeedType
          ]).


:- use_module(library(lists)).

:- use_module(library(debug)).

% :- debug(ground_terms).
:-   debug(assert_base).
/*

?- bottom_atom(A).
A = has_car(east1, car_11) ;
A = has_car(east1, car_12) ;
A = has_car(east1, car_13) ;
A = has_car(east1, car_14) ;
A = eastbound(east1) ;
A = short(car_14) ;
A = short(car_12) ;
A = closed(car_12) ;
A = long(car_13) ;
A = long(car_11) ;
A = open_car(car_14) ;
A = open_car(car_13) ;
A = open_car(car_11) ;
A = shape(car_14, rectangle) ;
A = shape(car_13, rectangle) ;
A = shape(car_12, rectangle) ;
A = shape(car_11, rectangle) ;
A = wheels(car_14, 2) ;
A = wheels(car_13, 3) ;
A = wheels(car_12, 2) ;
A = wheels(car_11, 2) ;
A = load(car_14, circle, 1) ;
A = load(car_13, hexagon, 1) ;
A = load(car_12, triangle, 1) ;
A = load(car_11, rectangle, 3).

*/


bottom_atom(Atom) :-
    '$aleph_sat_litinfo'(_A, _B, IDxTerm, _D, _E, _F),
    not(predicate_property(IDxTerm, built_in)), % Don't want to clash with built ins
    IDxTerm =.. [Pred | VarIds],
    idxs_to_groundterms(VarIds, GroundTerms),
    debug(ground_terms, 'Ground Terms: ~q', [GroundTerms]),
    Atom =..[Pred | GroundTerms],
    true.

/*
?- idxs_to_groundterms([2, aleph_const(rectangle), aleph_const(3)], G).
G = [car_11, rectangle, 3].
*/

idxs_to_groundterms([], []).
idxs_to_groundterms([aleph_const(GroundTerm) | Idxs], [GroundTerm | GroundTerms]):-
    !,
    idxs_to_groundterms(Idxs, GroundTerms).
idxs_to_groundterms([VarId | Idxs], [GroundTerm | GroundTerms]):-
    '$aleph_sat_terms'(VarId, _B2, GroundTerm, _Type),
    idxs_to_groundterms(Idxs, GroundTerms).



/*
import_base_via_saturation(+SeedAtom, +SeedType)

Import background predicates (as ground atoms) vi saturation.

Used when proving through an alterative database.

This leaves the user: namespace with (ground) atoms that were used to
form the bottom clause.

*/
import_base_via_saturation(SeedAtom, SeedType) :-
    memberchk(SeedType, [pos, neg, uspec]),  % TODO: Check that record_example/4 works as suspected
    record_example(check, SeedType, SeedAtom, ExIdx),
    sat(neg, ExIdx),
    !,
    bottom_set(SeedAtom, Base),
    assert_base(SeedAtom, Base),
    true.

/*
 ?- bottom_set(S,B).

S = eastbound(west10),
B = [has_car(west10, car_101), has_car(west10, car_102), short(car_101), long(car_102), open_car(car_102), open_car(car_101), shape(car_102, rectangle), shape(car_101, u_shaped), wheels(..., ...)|...] ;

*/
bottom_set(Seed, SeedHerbrandBase) :-
    bottom((Head :- _Body)),
    findall(Atom, bottom_atom(Atom), Atoms),
    select(Head, Atoms, SeedHerbrandBase),
    Seed = Head,
    true.

assert_base(Seed, []) :-
    debug(assert_base, 'Base Asserted for ~w', [Seed]).
assert_base(Seed, [Atom | Base]) :-
    debug(assert_base, 'Base atom to be asserted: ~q', [Atom]),
    (   user:clause(Atom , true, _Ref)
    ->  true  % Already exists -- no need to assert it again
    ;   user:asserta((Atom :- true)),   % Assert into module 'user'
        debug(assert_base, 'Base atom asserted: ~q', [Atom])
    ),
    update_base_register(Atom),
    assert_base(Seed, Base).


update_base_register(Atom) :-
    functor(Atom, Functor, Arity),
    (
    user:'$acuity_global'(imported_atom, Functor/Arity)
     -> true
     ; user:assert('$acuity_global'(imported_atom, Functor/Arity))
    ),
    true.


% show(imported)
%
% Lists the atoms that have been imported via external database and
% saturation of negative example.
%
% Need to add this in around line 4976 in aleph6.pl
%

user:show(imported) :-
    findall(FunctorArity,
            '$acuity_global'(imported_atom, FunctorArity),
            Atoms),
    format('[imported]~n', []),
    show_imported(Atoms),
    true.
user:show(imported) :-
    !.

show_imported([]).
show_imported([Functor/Arity | FAs]) :-
    functor(Atom, Functor, Arity),
    findall(Atom, clause(Atom, true, _Ref), Atoms),
    show_atoms(Atoms),
    show_imported( FAs).

show_atoms([]) :-
    format('~n').


show_atoms([Atom | Atoms]) :-
    format('~q. ~n', [Atom]),
    show_atoms( Atoms).


/* end of bottom_model.pl */



