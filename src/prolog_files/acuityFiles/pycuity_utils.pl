
/* pycuity_utils.pl
 * This returns a list given a clause. The hypothesis returned by Aleph is a compound term
 * This causes problems for the SwiplServer module. This file contains a predicate that converts
 * the hypothesis into a list of clauses.
 *   */


new_clause_list([_,_|NCL], NCL).
new_clause([NewClause|_], NewClause ). 

reduce_clause(BodyList, NewClause) :- new_clause_list(BodyList, NCL), new_clause(NCL, NewClause).
current_term([_,CurrentTerm|_], CurrentTerm).
insertAtEnd(X,Y,Z) :- append(Y,[X],Z).
variable_check([_,B|_]) :- var(B).
list_to_term([Functor|List], Term) :-
    Term =.. [Functor | List].

var_test([_,B|_]) :- var(B).
clause2list(Body , Lst, Output, BodyList) :- Body =.. BodyList, variable_check(BodyList), reverse(Lst,Output).
clause2list(Body, Lst, Output, ClauseOutput) :- Body =.. BodyList, 
                    reduce_clause(BodyList, NewClause), 
                    current_term(BodyList, CurrentTerm),
                    insertAtEnd(CurrentTerm, Lst, Lst1),
                    clause2list(NewClause, Lst1, Output, ClauseOutput).

bodyList(Body, FinalList) :- clause2list(Body,[],Output, Clause), list_to_term(Clause, Term), insertAtEnd(Term,Output,FinalList), !.

test_hypothesis(Head,Body,_) :- Head = true_class(A), Body = (contains(B,A), has_shape(B,cylinder), has_size(B,large), contains(C,A), has_shape(C,cube), left_of(B,C,A)).

% Write list elements to a file, each on a separate line
write_list_to_file(File, List) :-
    open(File, write, Stream),
    write_list_elements(Stream, List),
    close(Stream).

write_list_elements(_, []).
write_list_elements(Stream, [H|T]) :-
    write(Stream, H),
    nl(Stream), % Add a newline after each element
    write_list_elements(Stream, T).

% Predicate to append a list to a file
append_list_to_file(File, List) :-
    open(File, append, Stream, [alias(output)]),
    write(Stream, List),
    write(Stream, '.\n'),  % Add a newline to separate lists
    close(Stream).
