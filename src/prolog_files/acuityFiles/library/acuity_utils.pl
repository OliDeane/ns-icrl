/* acuity_utils.pl

    Author:        Steve Moyle
    Company:       Amplify Intelligence
    E-mail:        katarapkokid@hotmail.com
    Copyright (c)  2017
    All rights reserved.


Library to provide utility predictes for Acuity modules.
*/
:- module(acuity_utils, [
              list_with_index/2,   %+List, -WithIndex
              random_nitems_from_indexed_list/3 % +N, +IdxList, -RandomList
          ]).

:- use_module(library(lists)).
:- use_module(library(random)).

/*
?- list_with_index([a,b,c,d,e,f], I).
I = [1-a, 2-b, 3-c, 4-d, 5-e, 6-f].
    */
list_with_index(List, WithIndex) :-
    list_with_index(List, 1, WithIndex).

list_with_index([], _N, []).
list_with_index([L | List], N0, [N0-L | WithIndex]) :-
    N1 is N0 + 1,
    list_with_index(List, N1, WithIndex).

/*
?- db_extent:random_nitems_from_indexed_list(3,[1-a,2-b,3-c,4-d,5-e,6-f],R).
R = [5-e, 6-f, 4-d] ;
false.

?- db_extent:random_nitems_from_indexed_list(3,[1-a,2-b,3-c,4-d,5-e,6-f],R).
R = [2-b, 5-e, 3-c] .

?- db_extent:random_nitems_from_indexed_list(3,[1-a,2-b,3-c,4-d,5-e,6-f],R).
R = [3-c, 1-a, 5-e] .
*/
random_nitems_from_indexed_list(N, IdxList, RandomList) :-
    random_nitems_from_indexed_list(N, IdxList, [], RandomList).

random_nitems_from_indexed_list(0, _IdxList, RandomList, RandomList).
random_nitems_from_indexed_list(N, IdxList, RandomListIn, RandomL) :-
    random_member(IdxItem, IdxList),
    delete(IdxList, IdxItem, IdxList1),
    N1 is N - 1,
    random_nitems_from_indexed_list(N1, IdxList1, [IdxItem | RandomListIn], RandomL),
    true.








/* end of acuity_utils.pl */
