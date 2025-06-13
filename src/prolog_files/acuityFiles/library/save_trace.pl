/* save_trace.pl
 *
 *  Save the swipl console interaction trace.
 *
 */

:- module(save_trace, [
              save_trace/0,
              save_trace/1, % +FullFilePath
              trace_file/1
           ]).

:- use_module(library(debug)).

% :- debug(file_name).
:- debug(full_path).

save_trace :-
    trace_file(Filename),
    working_directory(CWD, CWD),
    format(atom(FullPath), '~w~w', [CWD, Filename]),
    debug(full_path, 'Protocol file: ~w', [FullPath]),
    save_trace(FullPath),
    true.

save_trace(Filename) :-
    protocol(Filename),
    true.

trace_file(Filename) :-
    get_time(TimeStamp),
    stamp_date_time(TimeStamp, DateTime, 'UTC'),
    format_time(atom(Filename),
                    'swipl-trace-%Y%m%d-%H%M.txt',
                    DateTime),
    debug(file_name, '~w', [Filename]),
    true.

 /* end of save_trace.pl */
