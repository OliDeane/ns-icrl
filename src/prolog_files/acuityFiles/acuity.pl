/* acuity.pl
 *  (C) 2016, 2017  Oliver Ray, Steve Moyle.
 *  @copyright Oliver Ray, Steve Moyle.
 *
 *  Note: This is a wrapper program to Aleph (version 5) and as such
 *  this wrapper is written in a compatible style. This means
 *  (almost) no modules and no exports.
 *
 *   @tbd
 *   1. Catch arithmetic calls that are unbound so that they can have
 *   predicates that bind or pass them by (automatically?)
 *
 *   2. Ensure that the added rebuttals and supporting infrastructure
 *   are reported in the final show(theory).
 *
 *   3. Add to the extent the number of elements in the list (for
 *   ease of understanding)
 *
 *   4. Implement a global reset of '$acuity_' and friends
 *
 *   5. Move the 'extent' functionality out of the Aleph loop
 *   and make it available via show(extent). (This is more
 *   Aleph idiomatic).
 *
 *   6. Replace show(Command) with something that does not need
 *   initialisation-time assertion. Perhaps acuity_show(Command) would
 *   be neater.
 *
 *   7.	When changing Skolem constants in rebuttal clauses, check that
 *   the negative example (once the head of the rebuttal) also has its
 *   skolem constant changed. This will involve some book keeping in
 *   the Aleph neg example structures.
 *
 *   */

:- use_module(library(gensym)).
% :- use_module(library(gui_tracer)).  % Comment this out if not used!
:- use_module(library(lists)).
:- use_module(library(pairs)).
:- use_module(library(varnumbers)).

:- dynamic(show_options/1).
:- dynamic(process_hypothesis/1).
:- dynamic(id/1).  % ? Int -- to keep track of Skolem variable numbering
:- dynamic(show/1).  % #universe
:- dynamic(prune/1). % prune


:- dynamic('$acuity_global'/2). %+Name, -Term

:- multifile(process_hypothesis/1).
:- multifile(show_options/1).
:- multifile(show/1).

:- discontiguous(process_hypothesis/1).
:- discontiguous(show_options/1). % show_options(hypothesis_selection)
:- discontiguous(show/1).

:- include('./Version6/aleph6.pl').

:- use_module('./library/db_proving/db_extent').
:- use_module('./library/search_shaping/search_shaping').

:- use_module(library(debug)).
% :- debug(rebut).
% :- debug(show_universe).
% :- debug(var_type).
% :- debug(alc).

:- initialization(reset_acuity).

reset_acuity :-
	format('TODO: Add Reset capability to ACUITY~nand trigger it at initialisation.~n~n~n', []), 
	reset_gensym('sk_'),
	reset_gensym('clause_'). % 

% The following definition of show_options(hypothesis_selection)
% is a replacement of the version in the original
% show_options(hypothesis_selection) in Aleph.
% It expands the original to have two new menu items:
% "extent" and "rebut"
% AssertA says put this in front of any other show_options clauses. AssertZ says put it after other show_options clauses. 



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


%%	skolemize(+ListOfVars, +IntegerID, -ListOfSkolemVars) is det
%
%	Produce unique ground Skolem constants based on an +IntegerID.
%
%	sk_2_3 denotes the 3rd skolem constant
%	(corresponding to variable "C" in the relevant hypothesis) in
%	the 2nd rebuttal issued.
%
%       ==
%       ?- skolemize([A, B, C, D], 6, S).
%       S = [sk_6_1, sk_6_2, sk_6_3, sk_6_4].
%       ==
% If you negate something that was for all A - then the negation is that there exists an X that does not. 
% Skolem - said we don;t know which element in our univers is the thing that causes negation. 
% So we make one up, to make the for all A negative. Different dollar sign to others. 
% d
skolemize(Vars, RebuttalID, Skolems) :-
	copy_term(Vars, NewVars),
	numbervars(NewVars),
	findall(Skolem,
		(
		    member('$VAR'(Num), NewVars),
		    NumPlusOne is Num+1,
		    atomic_list_concat(['sk_', RebuttalID, '_', NumPlusOne],
				       Skolem)
		), Skolems).

%%	assert_literals(','(+Literal, +Literals)) is det
%
%	Assert a rebutal =Body= using literals containing (existentially
%	quantified) variables with skolem constants.
%
assert_literals(true) :-
	debug(rebut,'No body literals to assert\n',[]),
	!.
assert_literals(','(Literal, Literals)) :-
	!,
	assert_literal(Literal),
	assert_literals(Literals).
assert_literals(Literal) :-
%	\+(Literal = ','(L, Ls) ; Literal = true) %  via cuts in prev
%	clauses
	assert_literal(Literal).

%%	assert_literal(+Literal) is det
%
%	Assert a rebutal =Body= literal containing (existentially
%	quantified) variables with skolem constants.
%
%       @bug Warning:  This is a crude and incomplete approach.
%	Lots of meta-interpreting or even maybe some CLP(FD) is
%	required.
%
assert_literal(\=(A,B)) :-		% inequalities are handled implicitly
	debug(rebut, 'Ignored ~q \\= ~q\n', [A, B]),
	!.
assert_literal(not(A)) :-		% negations are handled implicitly
	debug(rebut, 'Ignored not(~q)\n', [A]),
	!.
assert_literal(L) :-			% but these and other builtins may not work properly
	functor(L, Predicate, N),
	debug(var_type, 'Input literal L: ~q', [L]),
	skolem_lit_with_types(L, L1),
	debug(var_type, 'Skolemed literal L1: ~q', [L1]),
	length(Vars, N),
	M =..[Predicate | Vars],
	asserta(M :- (M == L1 -> !; M = L1), ClauseRef),
					% try to stop queries with skolems falling through
					% cf. gt1000(X):-(X=='$VAR'(0)->!;X='$VAR'(0)).
					% where gt1000('$VAR'(0)) succeeds and does not then cause a type error
					% but gt1000(X) returns '$VAR'(0) in addition to anything it would have done
					% (which is needed so Bottom Set is not accidentally pruned)
	get_new_id(clause, NewID),
	assert('$acuity_global'(rebuttal_clause_ref, NewID/ClauseRef)),
	(   debugging(rebut) -> instance(ClauseRef, Clause); true),
	debug(rebut, 'Asserted ~q\n',[Clause]),
	!. % Backtracking asserts way too many clauses


%%	skolem_lit_with_types(+LitIn, -LitOut) is nondet
%
%	Augement the rebut skolemised LitIn by appending plausible
%	Aleph type label to each skolem constant
%
%       ==
%       ?- skolem_lit_with_types(http_domain_name_parameter(sk_3_2,sk_3_1,sk_3_3,sk_3_5),
%	S = http_domain_name_parameter(sk_3_2_machine,
%	     sk_3_1_domain, sk_3_3_name, sk_3_5_parameter)
%       ==
%
skolem_lit_with_types(LitIn, LitOut) :-
	functor(LitIn, Predicate, N),
	LitIn =..[Predicate | Args],
	typed_vars(Predicate/N, Args, TypedArgs),
	debug(var_type, 'TypedArgs: ~q', [TypedArgs]),
	length(TypedArgs, N),
	LitOut =..[Predicate | TypedArgs],
	true.


%%	typed_vars(+Pred/+Arity, +SkolemConstList,
%	                              -SkolemConstWithTypeList) is det.
%	e.g.
%	==
%       SkolemConstList:
%	[sk_5_1,sk_5_4,sk_5_2,sk_5_5,'POST',sk_5_3,sk_5_6,sk_5_7,sk_5_8,
%	 sk_5_9,sk_5_10,sk_5_11,sk_5_12,sk_5_13,vector('text/plain'),
%        sk_5_14,vector('text/plain')]
%
%       SkolemConstWithTypeList:
%	[sk_5_1_time,sk_5_4_http_id,sk_5_2_machine,sk_5_5_ip,'POST',
%	 sk_5_3_domain,sk_5_6_uri_name,sk_5_7_uri_parameter,sk_5_8_referer,
%	 sk_5_9_user_agent,sk_5_10_size,sk_5_11_size,sk_5_12_response_code,
%	 sk_5_13_malware_request_uid,vector('text/plain'),
%        sk_5_14_malware_response_uid,vector('text/plain')]
%        ==
%
%        Note that existing constants are transferred into the TypedList
%
typed_vars(_Predicate/_Arity, [], []).
% SkVar is of the form sk_5_11 ...
typed_vars(Predicate/Arity, [SkVar | Vars], [TypedSkVar | TSVs]) :-
	rebut_skolem_var(SkVar),
	!,
	length(Vars, L),
	ArgNo is Arity - L,  % The current arg index for SkVar
	var_type(Predicate/Arity, ArgNo, BestType),
	atomic_list_concat([SkVar, '_', BestType], TypedSkVar),
	typed_vars(Predicate/Arity, Vars, TSVs).
% Const is some existing constant value we wish to keep
% e.g. 'POST', ..., vector('text/plain')
typed_vars(Predicate/Arity, [Const | Vars], [Const | TSVs]) :-
	% \+rebut_skolem_var(Const), % via the cut in the prev clause
	typed_vars(Predicate/Arity, Vars, TSVs).


%%	var_type(+Pred/+Arity, ArgNo, -BestType) is nondet
%
%       Use asserted Aleph modes for Pred/Arity to determine what the
%       BestType identifier to use for a variable at argument position ArgNo.
%
%	NOTE: There may be more than one mode type matching the
%	Pred/Arity template. In such a case there is a preference
%	ordering over the types at that ArgNo postion
%
%	* Precedence ordering constant '#' > input '+' > output '-'
%
%       ==
%       ?- var_type(http_domain_name_parameter/4, 2, T).
%            ArgNo: 2, Literal:
%	     http_domain_name_parameter(-machine,+domain,-name,-parameter),
%	     Value: +domain, PriF: +, Type: [domain]
%	   T = domain .
%
%	?- var_type(http_domain_name_parameter/4, 3, T).
%            ArgNo: 3, Literal:
%	     http_domain_name_parameter(-machine,+domain,-name,-parameter),
%	     Value: -name, PriF: -, Type: [name]
%	   T = name .
%       ==
%       @tbd Precedence ordering is unlikely to work
%
var_type(Pred/Arity, ArgNo, BestType) :-
	'$aleph_global'(modeh, modeh(_Recall, Literal)),
	functor(Literal, Pred, Arity),
	arg(ArgNo, Literal, Value),
	functor(Value, IOMarker, TypeArity),
	debug(var_type, 'ArgNo: ~q, Lit: ~q, Value: ~q', [ArgNo, Literal, Value]),
	arg(1, Value, TypeTerm), % 1st Arg is the TypeTerm
	debug(var_type, 'ArgNo: ~q, Literal: ~q, Value: ~q, IOMarker: ~q, TypeTerm: ~q',
	     [ArgNo, Literal, Value, IOMarker, TypeTerm]),
	(  \+ TypeArity == 1
	->
	    format('ERROR: Unable to assign types to skolem constants.~n'),
	    format('       Non atomic nor compound terms not supported ~w.~n', [TypeTerm]),
	    fail
	;
		(   IOMarker == (#)    % Precedence ordering '#' > '+' > '-'
		->  BestType = TypeTerm
		;   IOMarker == (+)
		->  BestType = TypeTerm
		;   IOMarker = (-)
		->  BestType = TypeTerm
		)
	).

%%	rebut_skolem_var(+SkVar) is det
%
%	Check if SkVar is a Skolem variable of the form 'sk_..'
%	already generated by the rebuttal process.
%
%       ==
%       ?- rebut_skolem_var(sk_4_1).
%       true.
%       ==
%
rebut_skolem_var(SkVar) :-
	atomic(SkVar),
	debug(alc, 'Prior to ALC call -- SkVar is: ~q', [SkVar]),
	atomic_list_concat(['sk' | _Rest], '_', SkVar).


%%	get_new_id(-NewID) is det
%
%	Generate an unique counter to ensure that generated Skolem
%	constants are unique.
%
get_new_id(NewID) :-
	get_new_id('sk', NewID).

%%	get_new_id(Base, -NewID) is det
%
%	Using the gensym framework, generate an unique counter to ensure
%	that generated Base constants are unique.
%
%	Beware: it uses the underscore '_' character as a separator
%	restricting base to NOT contain any.
%
get_new_id(Base, NewID) :-
	(   contains_underscore(Base)
	->  format('Warning: get_new_id/2 - Base must not contain an underscore. ~q was used.~n', [Base]),
	    fail
	;   (
		atomic_concat(Base, '_', Prefix),
		gensym(Prefix, SkTerm),
		atomic_list_concat([Base, AtomNewID], '_', SkTerm),
		atom_number(AtomNewID, NewID)
	    )
	).

contains_underscore(Atom) :-
	name(Atom, CharList),
	memberchk("_", CharList).

%    The following definitions extend the version in the
%    original process_hypothesis in Aleph.
%    It expands the original to have two new capabilities:
%    "extent" and "rebut"
% AAA SHOVING ing some more clauses into ALEPH. Adding different bindings. 
:- initialization(asserta(
(
process_hypothesis(extent):-
	!,
	hypothesis(Head, Body, _),
	asserta( Head :- Body, ID),
	(setof(Head, Head, Extent) -> true; Extent = []),
	format("Extent of proposed hypothesis is ~q", [Extent]),
	erase(ID),
	nl, p_message('done')
)
)).


:- initialization(asserta(
(
process_hypothesis(db_extent):-
	!,
	db_extent,
/*	hypothesis(Head, Body, _),
	asserta( Head :- Body, ID),
	(setof(Head, Head, Extent) -> true; Extent = []),
	format("Extent of proposed hypothesis is ~q", [Extent]),
	erase(ID),
*/
	nl, p_message('done')
)
)).
% How the menu button gets triggered. 
:- initialization(asserta(
(
process_hypothesis(rebut):-
	!,
	get_new_id(RebuttalID),
	hypothesis(Head, Body, _),
	term_variables(Head :- Body, Vars),
	skolemize(Vars, RebuttalID, Skolems),
	Vars = Skolems,
	assert_literals(Body),
	nl, p_message('added new rebuttal atoms and clauses'),
	skolem_lit_with_types(Head, Head1),
	debug(rebut,'denying ~q or ~q', [Head, Head1]),
	process_hypothesis(overgeneral because not(Head1))
)
)).


% The hypothesis ALEPH is currently about to score - can be the bottom clause or the thing currently being scored

:- initialization(asserta(
(
process_hypothesis(constrain):-
	!,
	hypothesis(Head, Body, _),
	search_shaping((Head :- Body), _),	
	nl, p_message('done')
)
)).

:- initialization(asserta(
(
process_hypothesis(pick):-
	!,
        bottom(Bottom),
	search_shaping(Bottom,_),
	nl, p_message('done')
)
)).

/*
?- catch(X is Y + 5, A, (debug)).
A = error(instantiation_error, context(system: (is)/2, _G28)).

[debug]  ?- catch(X is 2 + 5, A, (debug)).
X = 7.

Need to handle (amongst other things):
ERROR: =</2: Arithmetic: `sk_3_12/0' is not a function
   Exception: (24) index_prove1(1000/10000000000.0/sld, neg, (_G2152:-_G2152), 5, 5, _G2513) ? creep
   Exception: (17) get_gain1(set(5000, false, false, bf/coverage, false, 1, 10, false, false, false, false, false, 1, 1000, 3, 2, 0, 0.0, -10000000000.0, 0, 0.05, bf, [], 5, 10000000000.0, false, false, [], 10000000000.0, saturation, [1, 2, 3], 10000000000.0, false, sld, false, 10000000000.0, false, 10000000000.0, gsat, 0, swi, true, 1, false, 10000000000.0, false, 10000000000.0), upper, (malware_fetch(_G1214, _G1215, _G1216):-http(_G1214, _G1406, _G1215, _G1408, 'POST', _G1216, _G1411, _G1412, _G1413, _G1414, _G1415, _G1416, _G1417, _G1418, vector('text/plain'), _G1420, vector('text/plain')), http_category(_G1417, success)), 3, 0/3, 8, [0, 0, 0, -10000000000.0]/0, [1, 4], 6, [1-1, 2-2], [4-4, 5-5], [1, 2, 3, 7, 6, 8, 9, 10|...], 5, _G2521) ?


*/

user:prolog_exception_hook(
	 error(instantiation_error, _, _),
	 _, _, _) :-
	trace,  % TODO -- add some way of restarting the interactive session
	fail.



%%	show(universe) is det
%
%	Show the known Herbrand universe that has been 'discovered'
%	through the process of generating a bottom set.
/*
show(universe).

Show the known Herbrand universe that has been 'discovered' through the
process of generating a bottom set.

?- listing('$aleph_sat_terms').
:- dynamic'$aleph_sat_terms'/4.

'$aleph_sat_terms'(19, 2, success, category).
'$aleph_sat_terms'(18, 1, vector('text/plain'), mime_type).
'$aleph_sat_terms'(17, 1, 'POST', http_command).
'$aleph_sat_terms'(16, 1, vector('FwF43JMekCWMkW4xj'), malware_response_uid).
'$aleph_sat_terms'(15, 1, vector('F9DYM23Pa0DoRHXhs9'), malware_request_uid).
'$aleph_sat_terms'(14, 1, 200, response_code).
'$aleph_sat_terms'(13, 1, 123210, size).
'$aleph_sat_terms'(12, 1, 119, size).
'$aleph_sat_terms'(11, 1, 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.1; Trident/7.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0)', user_agent).
'$aleph_sat_terms'(10, 1, unset, referer).
'$aleph_sat_terms'(9, 1, okm0ua6s71c58, uri_parameter).
'$aleph_sat_terms'(8, 1, '/o51qYV.php', uri_name).
'$aleph_sat_terms'(7, 1, 'CHZkFF4L3Ey8GbJKV2', http_id).
'$aleph_sat_terms'(6, 1, ipv4(195, 208, 1, 122), ip).
'$aleph_sat_terms'(5, 1, 'CoPfcP251c0hXRko86', uid).
'$aleph_sat_terms'(4, 1, date(2015, 11, 5, 13, 4, 47, 55), time).
'$aleph_sat_terms'(3, 0, 'frc-conf.com', domain).
'$aleph_sat_terms'(2, 0, ipv4(192, 168, 122, 163), machine).
'$aleph_sat_terms'(1, 0, date(2015, 11, 5, 13, 5, 2, 506), time).

?- show(universe).

[Herbrand Universe]
category[1] - [success]
domain[1] - ['frc-conf.com']
http_command[1] - ['POST']
http_id[1] - ['CHZkFF4L3Ey8GbJKV2']
ip[1] - [ipv4(195,208,1,122)]
machine[1] - [ipv4(192,168,122,163)]
malware_request_uid[1] - [vector('F9DYM23Pa0DoRHXhs9')]
malware_response_uid[1] - [vector('FwF43JMekCWMkW4xj')]
mime_type[1] - [vector('text/plain')]
referer[1] - [unset]
response_code[1] - [200]
size[2] - [123210,119]
time[2] - [date(2015,11,5,13,4,47,55),date(2015,11,5,13,5,2,506)]
uid[1] - ['CoPfcP251c0hXRko86']
uri_name[1] - ['/o51qYV.php']
uri_parameter[1] - [okm0ua6s71c58]
user_agent[1] - ['Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.1; Trident/7.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0)']
true.

*/
%
show(universe) :-
	findall(Type-Value,
		'$aleph_sat_terms'(_Cntr, _Layer, Value, Type),
		Pairs),
	debug(show_universe, 'Findall: ~q', [Pairs]),
	keysort(Pairs, SortPairs),
	group_pairs_by_key(SortPairs, Group),
	debug(show_universe, 'Grouping: ~q', [Group]),
	p_message('Herbrand Universe'),
	forall((member(TypeKey-Values, Group), length(Values, LV)),
	       format('~q[~w] - ~q~n', [TypeKey, LV, Values])
	      ),
	true.


%%	show(rebuttals) is det
%
%	Show the asserted Skolemized rebuttal literals.
%
show(rebuttals) :-
	p_message('Rebuttal Clauses'),
	forall(rebuttal_clause(Clause, _ClauseRef),
	       pp_dclause(Clause)
	      ),
	true.
/*
?- listing('$acuity_global').
:- dynamic'$acuity_global'/2.

'$acuity_global'(rebuttal_clause_ref, 1/<clause>(0x27eed70)).
'$acuity_global'(rebuttal_clause_ref, 2/<clause>(0x27ef230)).
*/

rebuttal_clause(Clause, ClauseRef) :-
	'$acuity_global'(rebuttal_clause_ref, _ID/ClauseRef),
	instance(ClauseRef, Clause),
	true.


%% skolem_var_type(+SkolemVar, -Type)
%
%   Recover the Aleph -Type from the suffix of the +SkolemVar
%
%   ==
%   ?- skolem_var_type(sk_3_11_size, T).
%   T = size.
%
%   ?- skolem_var_type(sk_3_11_ip_address, T).
%   T = ip_address.
%   ==
%
skolem_var_type(SkVar, Type) :-
	atomic_list_concat([sk, _Num1, _Num2 | TypeList], '_', SkVar),
	atomic_list_concat(TypeList, '_', Type),
	true.

%% replace(+Old, +New,+ListOfTerms, -ListOfNewTerms).
%
%	replace atom +Old with atom +New inside the +ListOfTerms to
%	produce a new version -ListOfNewTerms.
%
%
%	==
%      ?- replace(sk_3_11_size, 100000, [
%	    (gt1000(_G242):-gt1000(_G242)==gt1000(sk_3_11_size)
%                    ->!;gt1000(_G242)=gt1000(sk_3_11_size))],
%                    [C]).
%	C = (gt1000(_G242):-gt1000(_G242)==gt1000(100000)
%                    ->!;gt1000(_G242)=gt1000(100000)).
%       ==
%
replace(_Old, _New, [], []).
replace(Old, New, [TermIn | InTerms], [TermOut | OutTerms]):-
    replace_term(Old, New, TermIn, TermOut),
    !,
    replace(Old, New, InTerms, OutTerms).
replace(Old, New, [TermInH| InTerms], [TermInH| OutTerms]):-
    replace(Old, New, InTerms, OutTerms).

%%   replace_term(+Old, +New, +TermIn, -TermOut)
%
replace_term(_Old, _New, InTerm, InTerm) :-
	var(InTerm), % Leave variables alone
	!.
replace_term(Old, New, InTerm, New) :-
	InTerm == Old,
	!.
replace_term(Old, New, InTerm, OutTerm) :- % +InTerm is a compound term
      InTerm =.. [F | OldArgs],  % Unpack the InTerm
      replace(Old, New, OldArgs, NewArgs),
      OutTerm =.. [F | NewArgs]. % Packup the OutTerm



%%	rebuttal(update)
%
%	Interactive precicate to update a stored rebuttal clause.
%
%	This displays possible clauses to be updated and the bottom
%	Herbrand universe as suggested replacements for Skolem
%	constants.
%
rebuttal(update) :-
	show(numbered_rebuttals),
	format('~n~t~4|Options:~n'),
	format('~t~8|-> Enter the [Number]. of the Rebuttal Clause to repair~n'),
	format('~t~8|-> \'none\'. to abort.~n'),
	repeat,
	read(RNum),
	( RNum = none
	->  format('~t~8| Aborting operation~n')
	;
	    '$acuity_global'(rebuttal_clause_ref, RNum/_CRef),
	     format('Handle clause [~w]~n', [RNum]),
	     ask_skolem_var_replacement(RNum)
	;
	    format('~t~8|Error: No Rebuttal with index [~q]~n', [RNum]),
	    format('~t~8|-> Enter the [Number] of the Rebuttal Clause to repair~n'),
	    fail
	),
	!,
	true.


ask_skolem_var_replacement(RNum):-
	show(universe),
	'$acuity_global'(rebuttal_clause_ref, RNum/CRef),
	instance(CRef, Clause),
	format('~n~t~4|Selected clause for replacement [~w]~n~t~4|~q', [RNum, Clause]),
	format('~n~t~4|Options:~n'),
	format('~t~8|-> Enter the [Skolem Constant]. to replace~n'),
	format('~t~8|-> \'none\'. to abort.~n'),
	repeat,
	read(SkVar),
	( SkVar = none
	->  format('~t~8| Aborting operation~n')
	;   format('~t~8| Skolem variable selected for replacement [~q]~n', [SkVar]),
	    skolem_var_type(SkVar, Type) ,
	    format('~t~8|  Consider selecting a universe value from the type [~q]~n', Type),
	    ask_replacement_skvar_value(RNum, SkVar)
	;   false
	),
	!,
	true.


ask_replacement_skvar_value(RNum, SkVar):-
	format('~n~t~4|Selected Skolem constant [~q] in clause [~q] for replacement~n', [SkVar, RNum]),
	format('~n~t~4|Options:~n'),
	format('~t~8|-> Enter the [Value]. to replace~n'),
	format('~t~8|-> \'none\'. to abort.~n'),
	repeat,
	read(Value),
	( Value = none
	->  format('~t~8| Aborting operation~n')
	;   format('~t~8| Replace [~q] with [~q] in clause [~q]~n', [SkVar, Value, RNum]),
	    retrieve_clause(RNum, InClause),
	    replace(SkVar, Value, [InClause], [OutClause]),
	    format('~t~8|Clause for repacement: ~q~n', [InClause]),
	    format('~t~8|Replacment Clause:     ~q~n', [OutClause]),
	    format('~t~8| \'ok.\' to continue, anything else to abort.~n'),
	    (   read(ok)
	    -> % gtrace,
	       replace_clause(RNum, OutClause)
	    ;   format('~t~8| Aborting operation~n')
	    )
	;   true
	),
	!,
	true.

%%	show(numbered_rebuttals) is det
%
%	Show the stored rebuttal clauses with their internal reference
%	number
%
:- initialization(asserta(
(
show(numbered_rebuttals) :-
		      !,
	format('[Rebuttal Clauses]~n', []),
	forall(	'$acuity_global'(rebuttal_clause_ref, RefTerm),
		(   RefTerm = RNum/ClauseRef,
		    instance(ClauseRef, Clause),
		    format('[~q] ~q~n', [RNum, Clause])
		)
	      )
))).

%%	retrieve_clause(+RNum, -Clause) is det
%
%	Retrieve a rebuttal -Clause stored at refernence +RNum
%
retrieve_clause(RNum, Clause) :-
	'$acuity_global'(rebuttal_clause_ref, RNum/CRef),
	instance(CRef, Clause),
	true.

%%	replace_clause(+RNum, +OutClause) is det
%
%	Replace the clause with index RNum with OutClause and update the
%	global reference.
%
%	NOTE: The +RNum is re-used for OutClause.
%
replace_clause(RNum, OutClause) :-
	'$acuity_global'(rebuttal_clause_ref, RNum/CRef),
	instance(CRef, _ExistingClause),
	asserta(OutClause, NewCRef),
	retract('$acuity_global'(rebuttal_clause_ref, RNum/CRef)),
	assert('$acuity_global'(rebuttal_clause_ref, RNum/NewCRef)),
	true.




bottom((Head :- Body)) :-
	'$aleph_sat'(lastlit,Last),
	get_clause(1,Last,[],[Head, BodyList]),
	list_to_goals(BodyList, Body).

/* Interactive mode we wish to see bottom regardless of verbosity */
show(bottom):-
	setting(interactive, true),
	nl,
	p_message('bottom clause'),
	'$aleph_sat'(lastlit,Last),
	get_clause(1,Last,[],FlatClause),
	pp_dlist(FlatClause).

/* end of acuity.pl */





