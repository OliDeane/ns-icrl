:- set(i,3).
:- set(verbose,2).
:- set(minpos, 2).
:- set(good,true).
:- set(noise,1).
% :- set(caching,true).

:- modeh(1, invalid(+sample)).
:- modeb(1, onRoad(+sample, #dir)).
:- modeb(1, beforeJunction(+sample,#dir)).
:- modeb(1, go(+sample, #dir)).
:- modeb(1, tls(+sample, #dir)).
:- modeb(1, tls(+sample, #dir)).
:- modeb(1, not(onRoad(+sample, #dir))).
:- modeb(1, not(beforeJunction(+sample,#dir))).

:- determination(invalid/1, onRoad/2).
:- determination(invalid/1, beforeJunction/2).
:- determination(invalid/1, go/2).
:- determination(invalid/1, tls/2).
:- determination(invalid/1, tls/2).
% :- determination(invalid/1,not/1).


row(N) :- between(0, 6, N).
col(N) :- between(0, 6, N).

dir(zero). dir(north). dir(south).
dir(east). dir(west).

go(S, zero) :- sample(S), dir(zero).  
go(S, north):- sample(S), dir(north).  
go(S, south) :- sample(S), dir(south).  
go(S, east) :- sample(S), dir(east).   
go(S, west) :- sample(S), dir(west).  

tls(0). tls(1).
tls(S, 0) :- sample(S), tls(0).
tls(S, 1) :- sample(S), tls(1). 

sample(sample).

at(S, C, R) :- sample(S), col(C), row(R).

onRoad(S, zero) :- at(S, X,Y), col(X), row(Y), X=3, Y=3.
onRoad(S, south) :- at(S, X,Y), col(X), row(Y), X=3, Y<3.
onRoad(S, north) :- at(S, X,Y), col(X), row(Y), X=3, Y>3.
onRoad(S, east) :- at(S, X,Y), col(X), row(Y), X>3, Y=3.
onRoad(S, west) :- at(S, X,Y), col(X), row(Y), X<3, Y=3.

beforeJunction(S, south) :- at(S, X,Y), col(X), row(Y), X=3, Y=2.
beforeJunction(S, north) :- at(S, X,Y), col(X), row(Y), X=3, Y=4.
beforeJunction(S, west) :- at(S, X,Y), col(X), row(Y), X=2, Y=3.
beforeJunction(S, east) :- at(S, X,Y), col(X), row(Y), X=4, Y=3.

% false :- hypothesis(_,B,_), goals_to_list(B,BodyList), !, \+ member(go(_,_),BodyList).
% false :- hypothesis(_,B,_), goals_to_list(B,BodyList), !, \+ member(onRoad(_,_),BodyList).

false :-
    hypothesis(_, B, _),
    goals_to_list(B, BodyList),
    member(beforeJunction(_, _), BodyList),
    \+ member(tls(_, _), BodyList).

:- consult('sumo_bidir_4way_junction2.bk').

% Generate sumo valid states
at(C,R) :- col(C), row(R).

write_list_to_file(File, List) :-
    open(File, write, Stream),
    write_list_elements(Stream, List),
    close(Stream).

write_list_elements(_, []).
write_list_elements(Stream, [H|T]) :-
    write(Stream, H),
    nl(Stream), % Add a newline after each element
    write_list_elements(Stream, T).


stateAction(C,R,Tls,A) :- at(C,R), tls(Tls), dir(A).
% generate_state_actions(AllValidStateActions) :- 
%     dynamic(sample/1),
%     dynamic(at/3),
%     dynamic(tls/2),
%     dynamic(go/2),
%     findall(
%         Vstates,
%         (
%             stateAction(C,R,Tls,A), 
%             assert(sample(state)), assert(at(state,C,R)), assert(tls(state,Tls)), assert(go(state, A)),
%             once((invalid2(state) -> fail ; true)),
%             retractall(sample(state)), retractall(at(state,C,R)), retractall(tls(state,Tls)), retractall(go(state, A)),
%             Vstates=[C,R,Tls,A]
%         ),
%         AllValidStateActions
%     ).

generate_state_actions(AllValid) :-
    dynamic(sample/1),
    dynamic(at/3),
    dynamic(tls/2),
    dynamic(go/2),
    findall([C,R,Tls,Act],
      ( stateAction(C,R,Tls,Act),
        setup_call_cleanup(
            ( assert(sample(state)),
              assert(at(state,C,R)),
              assert(tls(state,Tls)),
              assert(go(state,Act))
            ),
            invalid(state)     % keep only if invalid;
                                   %   drop the \+ if you want the invalid ones
            ,
            ( retract(sample(state)),
              retract(at(state,C,R)),
              retract(tls(state,Tls)),
              retract(go(state,Act))
            )
        )
      ),
      AllValid).

save_state_actions_to_file :-
    generate_state_actions(AllValidStateActions),
    % open('invalid_state_actions/new_violations2.txt', append, Stream),
    open('invalid_state_actions/new_violations.txt', append, Stream),
    forall(
        member(SubList, AllValidStateActions),
        (
            write(Stream, SubList),
            nl(Stream)
        )
    ),
    close(Stream).

% invalid(A) :- go(A,west), beforeJunction(A,east), tls(A,1).
% invalid(A) :- beforeJunction(B,south), go(B,west).
% invalid(A) :- go(C,east), beforeJunction(C,west), tls(C,1).

% invalid2(S) :- at(S,1,2).
% invalid2(A) :- onRoad(A,west), go(A,south).
% invalid2(A) :- onRoad(A,west), go(A,north).
% invalid2(A) :- onRoad(A,east), go(A,south).
% invalid2(A) :- onRoad(A,east), go(A,north).
% invalid2(A) :- onRoad(A,north), go(A,east).
% invalid2(A) :- onRoad(A,north), go(A,west).
% invalid2(A) :- onRoad(A,south), go(A,east).
% invalid2(A) :- onRoad(A,south), go(A,west).
% invalid2(A) :- beforeJunction(A,west), tls(A,1), go(A,east).
% invalid2(A) :- beforeJunction(A,east), tls(A,1), go(A,west).
% invalid2(A) :- beforeJunction(A,south), tls(A,0), go(A,north).
% invalid2(A) :- beforeJunction(A,north), tls(A,0), go(A,south).