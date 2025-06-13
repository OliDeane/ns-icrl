/* pycuity.pl
 *  (C) 2023  Oliver Deane, Oliver Ray, Steve Moyle.
 *  @copyright Oliver Deane, Oliver Ray, Steve Moyle.
 *
 *
 *   @tbd
 *   1. Initialization predicates for facilitating ALEPH's induce_incremental command through Python. 
 *   2. Return hypotheses and scores for incremental rule induction with ALEPH. 
 *
 *   */

% To operate the trace functionality. Run: save_facts_to_file("filename") outside of the interactive mode. The in the following session, run: restore_trace('filename') and rebuild_trace(A,B).


:- dynamic('$acuity_global'/2). %+Name, -Term
:- include('./acuity.pl').
:- include('./pycuity_utils.pl').
%:- use_module('./library/clause_selection.pl').
:- use_module('./library/acuity_utils').

:- use_module(library(http/json)).

% Suppress all warnings
:- set_prolog_flag(warn_singleton, false).
:- set_prolog_flag(verbose, silent).
:- set_prolog_flag(report_error, false).

:- initialization
       asserta(
(
show_options(hypothesis_selection):-
	   !, % Means that it won't try any other show_options below it. 
	   nl,
	   tab(4),
	   write('Options:'), nl,
	   tab(8),
	   write('-> "accept." to accept clause'), nl,
	   tab(8),
       write('-> "alternative_clauses." to explore other good clauses'), nl,
	   tab(8),
	   write('-> "prune." to prune clause and its refinements from the search'), nl,
	   tab(8),
	   write('-> "extent." to show the extent of the proposed hypothesis'), nl,
	   tab(8),
	   write('-> "db_extent." to show the extent of the proposed hypothesis'), nl,
	   tab(8),
	   write('                with respect to an external database'), nl,
	   tab(8),
	   write('-> "rebut." add rebuttal to the proposed hypothesis'), nl,
	   tab(8),
	   write('-> "constrain." add constaints from the proposed hypothesis'), nl,
	   tab(8),
	   write('-> "pick." add constaints from the most specific hypothesis'), nl,
	   tab(8),
	   write('-> "overgeneral." to add clause as a constraint'), nl,
	   tab(8),
	   write('-> "overgeneral because not(E)." to add E as a negative example'), nl,
	   tab(8),
	   write('-> "overspecific." to add clause as a positive example'), nl,
	   tab(8),
	   write('-> "overspecific because E." to add E as a positive example'), nl,
	   tab(8),
	   write('-> any Aleph command'), nl,
	   tab(8),
	   write('-> ctrl-D or "none." to end'), nl, nl
       )
).

% Induce_incremental options. We add this to aleph to allow for tracing and replay.
:- initialization(asserta(
    (
    process_hypothesis(prune):-
        !,
        retract('$aleph_global'(hypothesis,hypothesis(_,H,_,_))),
        Prune = (
                hypothesis(Head,Body,_),
                goals_to_list(Body,BodyL),
                clause_to_list(H,HL),
                aleph_subsumes(HL,[Head|BodyL])),
        assertz((prune(H):- Prune)),
        hypothesis(Hd, Bdy, _),
        user:asserta('$pycuity_user_trace'((Hd:-Bdy), prune(_, _, _, _))),
        nl, p_message('added new prune statement')
    )
    )).

:- initialization(asserta(
        (
        process_hypothesis(constrain):-
            !,
            hypothesis(Head, Body, _),
            search_shaping((Head :- Body), AcuityConstraintSpec),
            constraint_clause(_,_,Preds,MustOrNots,_,Index) = AcuityConstraintSpec,
            user:asserta('$pycuity_user_trace'((Head:-Body), constrain(MustOrNots, Preds, (Head:-Body), Index))),
            nl, p_message('done')
        )
        )).
        
:- initialization(asserta(
(
process_hypothesis(pick):-
    !,
    hypothesis(Head, Body, _),
    bottom(Bottom),
    search_shaping(Bottom,AcuityConstraintSpec),
    constraint_clause(_,_,Preds,MustOrNots,_,Index) = AcuityConstraintSpec,
    user:asserta('$pycuity_user_trace'((Head:-Body), pick(MustOrNots, Preds, Bottom, Index))), % We also save the Bottom clause to get the index on the trace rebuild. 
    nl, p_message('done')
)
)).

% This displays the response for the alternative_clauses option
% When a user selects the alternative_hypothesis_otion, they can select a hypothesis from the indexed list. 
% Note that the head of the add_to_hypothesis_clause/1 does not actually matter - we just care about the 
% indexes used in the clause body. I have not checked if al hypotheses can be successfully added. 
:- initialization(asserta(
(
process_hypothesis(alternative_clauses):-
	!,
    get_alt_clauses(GoodListWithIndex),
    alt_clause_selection(GoodListWithIndex),
	nl, p_message('done')
)
)).

alt_clause_selection(GoodListWithIndex) :-
    format('~n~t~4|Select alternative clause:~n'),
    format('~n~t~4|Clause Selection Options:~n'),
    print_good_clause_selection(GoodListWithIndex),
    format('~n~t~4| - Hypotheses Must contain clause \'add_to_hypothesis([i, .., n]).\'', []), 
    %format('~n~t~4| - Hypotheses Must NOT contain clause \'must_not([j, .., m]).\'', []),
    format('~n~t~8|-> Enter the add_to_hypothesis list. (e.g. |: add_to_hypothesis([1,1,5]). )'),
    format('~n~t~8|-> \'none\'. to abort.~n'),
    repeat,
    read(STerm),
    ( STerm = none
        ->  format('~t~8| Aborting operation~n')
    ;
    format('~n~t~8| [~w]~n', [STerm]),
    % STerm =.. [_,IDs],
    % p_message(IDs),
    add_selection_to_hypothesis(STerm, GoodListWithIndex),
    fail
    ),
    !,
    true.

add_selection_to_hypothesis(STerm, GoodListWithIndex) :-
     STerm =.. [_,IDs], 
     findall(SClause, (member(ID,IDs), member(ID-SClause,GoodListWithIndex), true), All_SClause),
     p_message(All_SClause),
     add_hyp_all(All_SClause),
%    PycuityAddedClauseSpec = addable_clause(LitsSpecs, SClause, ClauseListWithIndex, IDs),
    !,
    true.

get_alt_clauses(GoodListWithIndex) :- 
    findall(Good,'$aleph_good'(_,_,_,_,_,Good),GoodList),
    list_with_index(GoodList, GoodListWithIndex).    

print_good_clause_selection(GoodList) :-
    print_good_clauses(GoodList),
    true.

print_good_clauses([N-Lit]) :-
    !,  % Last element
    format('~t~4|[~w]     ~q.~n~n', [N, Lit]),
    true.
print_good_clauses([N-Lit | Lits]) :-
    format('~t~4|[~w]     ~q,~n', [N, Lit]),
    print_good_clauses(Lits).


% Base case: empty list, nothing to add_hyp to
add_hyp_all([]).

% Recursive case: apply add_hyp/1 to the head of the list,
% then recursively call add_hyp_all for the tail of the list
add_hyp_all([Clause|Rest]) :-
    add_hyp(Clause),
    add_hyp_all(Rest).

% Example usage
% All_SClause is your list of clauses
apply_add_hyp_to_all(All_SClause) :-
    Hyp = some_argument, % Provide the required argument for add_hyp/1
    add_hyp_all(All_SClause, Hyp).


% This reads in the pycuity_user_trace predicates stored in the user trace file. 
restore_trace(Filename) :-
    see(Filename),           % Open the file for reading
    repeat,
    read(Term),              % Read the next term from the file
    ( Term = end_of_file     % Stop if the end of file is reached
    ; assertz(Term),         % Assert the fact into the dynamic predicate
        fail                 % Continue reading
    ),
    seen,
    true.                    % Close the file

rebuild_trace(STerm, AcuityConstraintSpec) :- 
    '$pycuity_user_trace'(_,pick(MustOrNot, Preds, Clause, Index)),
    ConstraintTerm =.. [MustOrNot, Index], 
    search_shaping:clause_with_index(Clause, ClauseListWithIndex), search_shaping:check(ConstraintTerm, ClauseListWithIndex), 
    search_shaping:make_constraint_spec(ConstraintTerm,ClauseListWithIndex,AcuityConstraintSpec),
    assert(user:'$acuity_global'(constraint_spec, AcuityConstraintSpec), DBRef), fail. % Fail ensure this is run for all trace predicates


rebuild_trace(STerm, AcuityConstraintSpec) :- 
    '$pycuity_user_trace'(Clause,constrain(MustOrNot, Preds, _, Index)),
    ConstraintTerm =.. [MustOrNot, Index],
    p_message(ConstraintTerm),
    search_shaping:clause_with_index(Clause, ClauseListWithIndex), search_shaping:check(ConstraintTerm, ClauseListWithIndex), 
    search_shaping:make_constraint_spec(ConstraintTerm,ClauseListWithIndex,AcuityConstraintSpec),
    assert(user:'$acuity_global'(constraint_spec, AcuityConstraintSpec), DBRef),
    fail. % Fail ensure this is run for all trace predicates



% Define a predicate to save facts to a file
% Note the brackets around the first argument. 
% Define a FileName and the Predicate argument to save_facts_to_file/2
save_facts_to_file(FileName) :-
    open(FileName, write, Stream),
    forall(call('$pycuity_user_trace', User, Action),
           format(Stream, '~q((~w), ~w).~n', ['$pycuity_user_trace', User, Action])
          ),
    close(Stream). 

% This reads in the pycuity_user_trace predicates stored in the user trace file. 
restore_trace(Filename) :-
    see(Filename),           % Open the file for reading
    repeat,
    read(Term),              % Read the next term from the file
    ( Term = end_of_file     % Stop if the end of file is reached
    ; assertz(Term),         % Assert the fact into the dynamic predicate
        fail                    % Continue reading
    ),
    seen,
    true.                    % Close the file

save_trace(File) :-
    aleph_open(File,write,Stream),
    set_output(Stream),
    '$aleph_global'(rules,rules(L)),
    aleph_reverse(L,L1),
    write_rule(L1),
    flush_output(Stream),
    set_output(user_output).

get_hypothesis_as_list(TheoryList) :- 
    findall(
        Atom, 
        ('$aleph_global'(theory,theory(_,_,Clause,_,_)), 
        compound(Clause), 
        Clause = (Head :- _),
        bodyList(Clause, BodyList), Atom=BodyList), TheoryList).

fetch_hypothesis_with_aleph_method(OutputClause) :-
    findall(
        TempClause,
        ('$aleph_global'(rules,rules(L)), % This gives list opf rule indices.
        aleph_reverse(L,L1),
        aleph_member(ClauseNum,L1),
        '$aleph_global'(theory,theory(ClauseNum,_,_,_,_)),
        ClauseNum > 0, '$aleph_global'(theory,theory(ClauseNum,_,TempClause,_,_)))
        , OutputClause), get_head(TheoryList,ClauseHead, ClauseTail).

fetch_hypothesis_with_aleph_method2(OutputClause) :-
    findall(
        TempClauseString,
        ('$aleph_global'(rules,rules(L)), % This gives list opf rule indices.
        aleph_reverse(L,L1),
        aleph_member(ClauseNum,L1),
        '$aleph_global'(theory,theory(ClauseNum,_,_,_,_)),
        ClauseNum > 0, '$aleph_global'(theory,theory(ClauseNum,_,TempClause,_,_)),
        term_string(TempClause,TempClauseString))
        , OutputClause), get_head(TheoryList,ClauseHead, ClauseTail).

get_head([Head|Tail], Head, Tail).

fetch_clause(Clause):-
    hypothesis(H,B,_), RClause = (H:-B),% bodyList(RClause, RClauseList),
    term_string(RClause,Clause).

get_hypothesis_head(Head) :- 
    '$aleph_global'(theory,theory(_,_,Clause,_,_)), 
    compound(Clause), 
    Clause = (Head :- _).

initialise_incremental() :-
    clean_up,
    retractall('$aleph_global'(search_stats,search_stats(_,_))),
    store_values([interactive,portray_search,proof_strategy,mode]),
    set(portray_search,false),
    set(proof_strategy,sld),
    set(interactive,true),
    record_settings.

initialise_shaping_example(E,N) :-
    once(record_example(check,pos,E,N)),
    retractall('$aleph_global'(example_selected,
                    example_selected(_,_))),
    asserta('$aleph_global'(example_selected,
                    example_selected(pos,N))).

get_indexed_bottom_clause(ClauseList, BodyListWithIndex) :-
    numbervars(ClauseList, 0, _),
    ClauseList = [0-Head | BodyListWithIndex],
    format('~n~t~4|~q :-~n', [Head]),
    true.

% This searches for best clause for a single example (i.e., run on the reduce command)
% I have edited this so that it returns the best clause
find_clause(Search, RClause, RClauseList):-
    set(stage,reduction),
    set(searchstrat,Search),
    p_message('reduce'),
    reduce_prelims(L,P,N),
    asserta('$aleph_search'(openlist,[])),
    get_search_settings(S),
    arg(4,S,_/Evalfn),
    get_start_label(Evalfn,Label),
    ('$aleph_sat'(example,example(Num,Type)) ->
        '$aleph_example'(Num,Type,Example),
        asserta('$aleph_search'(selected,selected(Label,(Example:-true),
                            [Num-Num],[])));
        asserta('$aleph_search'(selected,selected(Label,(false:-true),[],[])))),
    arg(13,S,MinPos),
    interval_count(P,PosLeft),
    PosLeft >= MinPos,
    '$aleph_search'(selected,selected(L0,C0,P0,N0)),
    add_hyp(L0,C0,P0,N0),
        ('$aleph_global'(max_set,max_set(Type,Num,Label1,ClauseNum))->
        BestSoFar = Label1/ClauseNum;
        ('$aleph_global'(best,set(best,Label2))->
            BestSoFar = Label2/0;
            BestSoFar = Label/0)),
        asserta('$aleph_search'(best_label,BestSoFar)),
    p1_message(1,'best label so far'), p_message(1,BestSoFar),
        arg(3,S,RefineOp),
    stopwatch(StartClock),
        (RefineOp = false ->
                get_gains(S,0,BestSoFar,[],false,[],0,L,[1],P,N,[],1,Last,NextBest),
        update_max_head_count(0,Last);
        clear_cache,
        interval_count(P,MaxPC),
        asserta('$aleph_local'(max_head_count,MaxPC)),
        StartClause = 0-[Num,Type,[],false],
                get_gains(S,0,BestSoFar,StartClause,_,_,_,L,[StartClause],
                P,N,[],1,Last,NextBest)),
        asserta('$aleph_search_expansion'(1,0,1,Last)),
    get_nextbest(S,_),
    asserta('$aleph_search'(current,current(1,Last,NextBest))),
    search(S,Nodes),
    stopwatch(StopClock),
    Time is StopClock - StartClock,
        '$aleph_search'(selected,selected(BestLabel,RClause,PCover,NCover)),
    retract('$aleph_search'(openlist,_)),
    add_hyp(BestLabel,RClause,PCover,NCover),
    p1_message(1,'clauses constructed'), p_message(1,Nodes),
    p1_message(1,'search time'), p_message(1,Time),
    p_message(1,'best clause'),
%    pp_dclause(RClause), bodyList(RClause,RClauseList),
    pp_dclause(RClause), hypothesis(HeadA, BodyA,_), 
    goals_to_list(BodyA,BodyListA), goals_to_list(HeadA,HeadListA),
    append(HeadListA, BodyListA, RClauseList),
    p_message("Hello"), p_message(RClause), p_message(RClauseList),
    show_stats(Evalfn,BestLabel),
    update_search_stats(Nodes,Time),
    record_search_stats(RClause,Nodes,Time),
    noset(stage),
    !.


pick_with_python(Clause, STerm) :-
    search_shaping:clause_with_index(Clause, ClauseListWithIndex),
    % repeat,
    search_shaping:check(STerm, ClauseListWithIndex),
    search_shaping:make_constraint_spec(STerm,ClauseListWithIndex,AcuityConstraintSpec),
    % format('~n~t~8|Proposed constraint is: ~n', []),
    format('~n~t~8|     ~q~n', [AcuityConstraintSpec]),
    % format('~n~t~8|-> \'ok.\' to accept constraint or \'none.\' to abort.~n'),
    % read(Action),
    % (   Action = ok
    assert(user:'$acuity_global'(constraint_spec, AcuityConstraintSpec), DBRef),
    debug(constraint_spec, 'Constraint added at ref:  ~q.', [DBRef]),
    p_message(AcuityConstraintSpec),
    !,
    true.

% For removing specific, or all constraints. To remove all, pass in _,_. 
clear_constraints(Predicates,MustOrNot) :- 
    retractall('$acuity_global'(constraint_spec,constraint_clause(_,true,[Predicates],MustOrNot,_,_))),
    retractall('$pycuity_constraint_violations'(_)).

% Fetch the existing constraints
fetch_constraints(Constraints, MustOrNots) :- 
    findall(Predicates, '$acuity_global'(constraint_spec,constraint_clause(_,true,[Predicates],_,_,_)), Constraints),
    findall(Ms, '$acuity_global'(constraint_spec,constraint_clause(_,true,_,Ms,_,_)), MustOrNots).


% Fecth the search space and accompanying constraints/constraint violations
% fetch_search_space_constraints(Search_space, Constraint_violations) :-
%     findall(X, '$pycuity_search_store'(X), Ss),
%     findall(Y, '$pycuity_constraint_violations'(Y,_,_), Cv),
%     apply_term_string(Ss, Search_space),
% 	apply_term_string(Cv, Constraint_violations).


% This allows for extraction of the Cv that has been found
fetch_search_space_constraints(Search_space, Constraint_violations, MustOrNots, Constraint_predicates) :-
    findall(X, '$pycuity_search_store'(X), Ss),
    findall(Y, '$pycuity_constraint_violations'(Y,_,_), Cvs),
    findall(MoN, '$pycuity_constraint_violations'(_,MoN,_), MoNs),
    findall(Cp, '$pycuity_constraint_violations'(_,_,Cp), Cps),
    apply_term_string(Ss, Search_space),
	apply_term_string(Cvs, Constraint_violations),
    apply_term_string(MoNs, MustOrNots),
    apply_term_string(Cps, Constraint_predicates).


% Store ALEPH output in a json file
save_json_object(FilePath, Search_space, Constraint_violations) :-
    % Create a JSON object
    JSON = json{
        search_space: Search_space,
        constraint_violations: Constraint_violations
    },

    % Open the file in write mode
    open(FilePath, write, Stream),

    % Convert the JSON object to a string
    json_write(Stream, JSON, [width(0)]),

    % Close the file
    close(Stream).

save_search_constraints(Filename) :- 
	fetch_search_space_constraints(Ss, Cv),
	apply_term_string(Ss, Search_space),
	apply_term_string(Cv, Constraint_violations),
	save_json_object(Filename,Search_space, Constraint_violations).


% Convert compiunds in the list to strings to enable storage in json
apply_term_string([], []).
apply_term_string([Term|Rest], [String|StringRest]) :-
    term_string(Term, String),
    apply_term_string(Rest, StringRest).

% This is the clause used to display when a clause is found. Use this to shtore clause options.
% show_clause(Flag,Label,Clause,Nodes):-
%     broadcast(clause(Flag,Label,Clause,Nodes)),
% 	p_message('-------------------------------------'),
% 	(Flag=good -> p_message('good clause');
% 		(Flag=sample-> p_message('selected from sample');
% 			p_message('found clause'))),
% 	pp_dclause(Clause),
% 	(setting(evalfn,Evalfn)-> true; Evalfn = coverage),
% 	show_stats(Evalfn,Label),
% 	p1_message('clause label'), p_message(Label),
% 	p1_message('clauses constructed'), p_message(Nodes),
% 	p_message('-------------------------------------').
