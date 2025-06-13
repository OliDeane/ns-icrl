:-consult("../acuity.pl").

ask_example2(E):-
	('$aleph_global'(example_selected,example_selected(pos,N)) ->
		'$aleph_example'(N,pos,E1);
		E1 = none),
	!,
	show_options(example_selection),
	tab(4),
	write('Response '), p1_message(default:E1), write('?'), nl,
	read(Response),
	(Response = ok  -> E = E1; E = Response).

process_hypothesis2:-
	show(hypothesis),
	repeat,
	show_options(hypothesis_selection),
	tab(4),
	write('Response?'), nl,
	read(Response),
	process_hypothesis(Response),
	(Response = end_of_file; Response = none), !.

induce2(incremental):-
	clean_up,
	retractall('$aleph_global'(search_stats,search_stats(_,_))),
	store_values([interactive,portray_search,proof_strategy,mode]),
	set(portray_search,false),
	set(proof_strategy,sld),
	set(interactive,true),
	record_settings,
        stopwatch(StartClock),
        repeat,
		('$aleph_global'(example_selected,example_selected(pos,N)) ->
		'$aleph_example'(N,pos,E1);
		E1 = none),
	!,
	show_options(example_selection),
	tab(4),
	write('Response '), p1_message(default:E1), write('?'), nl,
	read(Response_atom), Response = eastbound(Response_atom),
	(Response = ok  -> E = E1; E = Response),
	((E = end_of_file; E = none) -> true;
		once(record_example(check,pos,E,N)),
		retractall('$aleph_global'(example_selected,
						example_selected(_,_))),
		asserta('$aleph_global'(example_selected,
						example_selected(pos,N))),
		once(sat(N)),
		once(reduce),
		once(process_hypothesis),
		fail),
	!,
        stopwatch(StopClock),
        Time is StopClock - StartClock,
        show(theory),
	show(pos),
	show(neg),
	show(false/0),
	show(prune/1),
        record_theory(Time),
	reinstate_values([interactive,portray_search,proof_strategy,mode]),
        p1_message(1,'time taken'), p_message(1,Time).

induce3(incremental):-
	clean_up,
	retractall('$aleph_global'(search_stats,search_stats(_,_))),
	store_values([interactive,portray_search,proof_strategy,mode]),
	set(portray_search,false),
	set(proof_strategy,sld),
	set(interactive,true),
	record_settings,
		stopwatch(StartClock),
		repeat,
	ask_example2(E),
	((E = end_of_file; E = none) -> true;
		once(record_example(check,pos,E,N)),
		retractall('$aleph_global'(example_selected,
						example_selected(_,_))),
		asserta('$aleph_global'(example_selected,
						example_selected(pos,N))),
		once(sat(N)),
		once(reduce),
		once(process_hypothesis),
		fail),
	!,
		stopwatch(StopClock),
		Time is StopClock - StartClock,
		show(theory),
	show(pos),
	show(neg),
	show(false/0),
	show(prune/1),
		record_theory(Time),
	reinstate_values([interactive,portray_search,proof_strategy,mode]),
		p1_message(1,'time taken'), p_message(1,Time).

