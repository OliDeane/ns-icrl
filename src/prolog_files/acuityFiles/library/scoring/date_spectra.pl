/* date_spectra.pl

TODO

*/

:- module(date_spectra, [
              one_d_epsilon_clustering/3,   % +[SortedListOfNumbers], +Epsilon, -[Clusters]
              segment_timestamps/3, % +[SortedKVPairs], +Epsilon, -[Segments]
              clusters_statistics/2, % +[Clusters], -Stats
              minutes/2, % +[DateTimes], -[Minutes]
              hours/2, % +[DateTimes], -[Hours]
          ]).

:- use_module(library(lists)).

:- use_module(library(debug)).

:- nodebug(cluster_split).

/* Extract clock-face minutes

?- Ds = [date(2021, 1, 1, 0, 10, 1.917999982, 0, 'UTC', -), date(2021, 1, 1, 1, 10, 1.196000099, 0, 'UTC', -), date(2021, 1, 1, 2, 10, 1.5620000360000001, 0, 'UTC', -), date(2021, 1, 1, 3, 10,  1.825999975, 0, 'UTC', -), date(2021, 1, 1, 4, 10, 1.525000095, 0,'UTC', -), date(2021, 1,
1, 5, 10, 2.138999938, 0, 'UTC', -),  date(2021, 1, 1, 6, 10, 1.492000102, 0, 'UTC', -), date(2021, 1, 1, 7, 10, 1.384000062, 0, 'UTC', -)], minutes(Ds, Ms), one_d_epsilon_clustering(Ms, 0.5, Cs), clusters_statistics(Cs, Stats).
Correct to: "date_spectra:minutes(Ds,Ms)"? yes
Ds = [date(2021, 1, 1, 0, 10, 1.917999982, 0, 'UTC', -), date(2021, 1, 1, 1, 10, 1.196000099, 0, 'UTC', -), date(2021, 1, 1, 2, 10, 1.5620000360000001, 0, 'UTC', -), date(2021, 1, 1, 3, 10, 1.825999975, 0, 'UTC', -), date(2021, 1, 1, 4, 10, 1.525000095, 0, 'UTC', -), date(2021, 1, 1, 5, 10, 2.138999938, 0, 'UTC', -), date(2021, 1, 1, 6, 10, 1.492000102, 0, 'UTC', -), date(2021, 1, 1, 7, 10, 1.384000062, 0, 'UTC', -)],
Ms = [10.031966666366667, 10.019933334983333, 10.026033333933333, 10.030433332916667, 10.02541666825, 10.035649998966667, 10.024866668366666, 10.0230666677],
Cs = [[10.031966666366667, 10.019933334983333, 10.026033333933333, 10.030433332916667, 10.02541666825, 10.035649998966667, 10.024866668366666, 10.0230666677]],
Stats = [10.027170833935415/0.004801639068883972/8] ;


*/
minutes([], []).
minutes([DT | DateTimes], [Min | Minutes]) :-
    DT = date(_Year, _Month, _Day, _Hour, Mins, Seconds, _Offset, _TZ, _),
    Min is Mins + Seconds / 60.0,
    minutes(DateTimes, Minutes).

/*
Extract clock-face hours

?- Ds = [date(2021, 1, 1, 0, 10, 1.917999982, 0, 'UTC', -), date(2021, 1, 1, 1, 10, 1.196000099, 0, 'UTC', -), date(2021, 1, 1, 2, 10, 1.5620000360000001, 0, 'UTC', -), date(2021, 1, 1, 3, 10,  1.825999975, 0, 'UTC', -), date(2021, 1, 1, 4, 10, 1.525000095, 0,'UTC', -), date(2021, 1,
1, 5, 10, 2.138999938, 0, 'UTC', -), date(2021, 1, 1, 6, 10,
1.492000102, 0, 'UTC', -), date(2021, 1, 1, 7, 10, 1.384000062, 0,
'UTC', -)], hours(Ds, Hs), one_d_epsilon_clustering(Hs, 0.5, Cs),
clusters_statistics(Cs, Stats). Correct to: "date_spectra:hours(Ds,Hs)"?
yes
Ds = [date(2021, 1, 1, 0, 10, 1.917999982, 0, 'UTC', -), date(2021,1, 1, 1, 10, 1.196000099, 0, 'UTC', -), date(2021, 1, 1, 2, 10,
1.5620000360000001, 0, 'UTC', -), date(2021, 1, 1, 3, 10, 1.825999975,0, 'UTC', -), date(2021, 1, 1, 4, 10, 1.525000095, 0, 'UTC', -),
date(2021, 1, 1, 5, 10, 2.138999938, 0, 'UTC', -), date(2021, 1, 1, 6,10, 1.492000102, 0, 'UTC', -), date(2021, 1, 1, 7, 10, 1.384000062, 0,
'UTC', -)],
    Hs = [0.17199444439444445, 1.169988889163889,
2.1710055556555554, 3.171738888819444, 4.1709027780416665,
5.1726083331611115, 6.170811111394444, 7.170511111283334],
    Cs =[[0.17199444439444445], [1.169988889163889], [2.1710055556555554],
[3.171738888819444], [4.1709027780416665], [5.1726083331611115],
[6.170811111394444], [7.170511111283334]],
    Stats =[0.17199444439444445/0.0/1, 1.169988889163889/0.0/1,
2.1710055556555554/0.0/1, 3.171738888819444/0.0/1,
4.1709027780416665/0.0/1, 5.1726083331611115/0.0/1,
6.170811111394444/0.0/1, 7.170511111283334/0.0/1] .

?-
*/
hours([], []).
hours([DT | DateTimes], [Hrs | Hours]) :-
    DT = date(_Year, _Month, _Day, Hour, Mins, Seconds, _Offset, _TZ, _),
    Hrs is Hour + Mins/60.0 + Seconds / 360.0,
    hours(DateTimes, Hours).

/*
%  Ds = [date(2021, 1, 1, 0, 10, 1.917999982, 0, 'UTC', -), date(2021, 1, 1, 1, 10, 1.196000099, 0, 'UTC', -), date(2021, 1, 1, 2, 10, 1.5620000360000001, 0, 'UTC', -), date(2021, 1, 1, 3, 10,  1.825999975, 0, 'UTC', -), date(2021, 1, 1, 4, 10, 1.525000095, 0,'UTC', -), date(2021, 1, 1, 5, 10, 2.138999938, 0, 'UTC', -),  date(2021, 1, 1, 6, 10, 1.492000102, 0, 'UTC', -), date(2021, 1, 1, 7, 10, 1.384000062, 0, 'UTC', -)].
*/
dates_to_stamps([], []).
dates_to_stamps([Date | Dates], [Stamp | Stamps]) :-
    date_time_stamp(Date, Stamp),
    dates_to_stamps(Dates, Stamps).

stamps_deltas([_], []).
stamps_deltas([S1, S2 | Ss], [D12 | Ds]) :-
    D12 is S1 - S2,
    stamps_deltas([S2 | Ss],  Ds).


histogram(Bins, Data, Hist) :-
    assertion(sort(Bins, Bins)),
    initialise_bins(Bins, Bins0),
    histogram_(Bins, Data, Bins0, UnsortedHist),
    sort(UnsortedHist, Hist).

histogram_(_Bins, [], Hist, Hist) :-
    !.
histogram_(Bins, [D  |Data], Hist0, Hist) :-
    update_histogram(Bins, D, Hist0, Hist1),
    histogram_(Bins, Data, Hist1, Hist).

update_histogram(Bins, Data, Hist0, Hist) :-
    (   select_bin(Bins, Data, Bin)
    ->  member(Bin-Count0, Hist0),
        delete(Hist0, Bin-Count0, Hist1),
        Count1 is Count0 + 1,
        append([Bin-Count1], Hist1, Hist)
    ;   Hist = Hist0
    ).

select_bin([Bin0, Bin1 | _Bins], Data, Bin0) :-
    Data >= Bin0,
    Data < Bin1,
    !.
select_bin([LastBin], Data, LastBin) :-
    Data >= LastBin,
    !.
select_bin([FirstBin|_], Data, FirstBin) :-
    Data < FirstBin,
    format('Warning: data value (~w) less than smallest bin value (~w). Discarding. (Hint: add another smaller bin.)~n', [Data, FirstBin]),
    !,
    fail.
select_bin([Bin0, Bin1 | Bins], Data, Bin) :-
    Data >= Bin0,
    \+ Data < Bin1,
    select_bin([Bin1 | Bins], Data, Bin).

initialise_bins([], []).
initialise_bins([Bin | Bins], [Bin-0 | Bins0]) :-
    initialise_bins(Bins, Bins0).


/*
https://stackoverflow.com/questions/18364026/clustering-values-by-their-proximity-in-python-machine-learning

See: Don't use clustering for 1-dimensional data
*/


one_d_epsilon_clustering(Nums, Epsilon, Clusters) :-
    list_keyed_elems(Nums, KeyedElements),
    segment_timestamps(KeyedElements, Epsilon, KeyedElementClusters),
    strip_keys_from_clusters(KeyedElementClusters, Clusters).


strip_keys_from_clusters([], []).
strip_keys_from_clusters([KEC | KeyedElementClusters], [C | Clusters]) :-
    strip_keys(KEC, C),
    strip_keys_from_clusters(KeyedElementClusters, Clusters).


strip_keys([], []).
strip_keys([_Key-Elem | KeyedElementClusters], [Elem | Clusters]) :-
    strip_keys(KeyedElementClusters, Clusters).


clusters_statistics([], []).
clusters_statistics([C | Clusters], [CStat | Stats]) :-
    cluster_statistics(C, CStat),
    clusters_statistics(Clusters, Stats).

cluster_statistics(Cluster, Stats) :-
    length(Cluster, N),
    arithmetic_mean(Cluster, Mean),
    standard_deviation(Cluster, Mean, SD),
    Stats= Mean/SD/N,
    true.


/*
See: https://en.wikipedia.org/wiki/Standard_deviation

?- Xs = [2,4,4,4,5,5,7,9], arithmetic_mean(Xs, M).
Correct to: "date_spectra:arithmetic_mean(Xs,M)"? yes
Xs = [2, 4, 4, 4, 5, 5, 7, 9],
M = 5.

?- Xs = [2,4,4,4,5,5,7,9], arithmetic_mean(Xs, M), standard_deviation(Xs, M, SD).
Correct to: "date_spectra:arithmetic_mean(Xs,M)"? yes
Correct to: "date_spectra:standard_deviation(Xs,M,SD)"? yes
Xs = [2, 4, 4, 4, 5, 5, 7, 9],
M = 5,
SD = 2.0.

?-
*/
standard_deviation(Xs, Mean, SD) :-
    deviation_squares(Xs, Mean, DSquares),
    arithmetic_mean(DSquares, SigmaSquared),
    SD is SigmaSquared ^ 0.5,
    true.


deviation_squares([], _Mean, []).
deviation_squares([Xi | Xs], Mu, [Di2 |DSquares]) :-
    Di2 is (Xi - Mu) ^ 2,
    deviation_squares(Xs, Mu, DSquares).


arithmetic_mean(Values, Mean) :-
    sum_list_n(Values, Sum, N),
    Mean is Sum / N.

sum_list_n(Values, Sum, N) :-
    sum_list_n_(Values, 0, Sum, 0, N).

sum_list_n_([], Sum, Sum, N, N).
sum_list_n_([V | Values], Sum0, Sum, N0, N) :-
    Sum1 is Sum0 + V,
    N1 is N0 + 1,
    sum_list_n_(Values, Sum1, Sum, N1, N).


/*
[trace]  ?- split_list([1,2,3,4,5], [3], Ls).
Correct to: "date_spectra:split_list([1,2,3,4,5],[3],Ls)"? yes
Ls = [[1, 2, 3], [4, 5]] ;
false.

?- split_list([1,2,3,4,5], [2,1], Ls).
Correct to: "date_spectra:split_list([1,2,3,4,5],[2,1],Ls)"? yes
Ls = [[1, 2], [3], [4, 5]] .

?- split_list([1,2,3,4,5], [2,1,1], Ls).
Correct to: "date_spectra:split_list([1,2,3,4,5],[2,1,1],Ls)"? yes
Ls = [[1, 2], [3], [4], [5]] ;




    */
split_list(L, [], [L]).
split_list(L0, [Idx | SplitIdxs], [Prefix | SubLists]) :-
    length(Prefix, Idx),
    append(Prefix, L1, L0),
    split_list(L1, SplitIdxs, SubLists).



/*

From: https://stackoverflow.com/questions/11513484/1d-number-array-clustering

```python
clusters = []
eps = 0.2
points_sorted = sorted(points)
curr_point = points_sorted[0]
curr_cluster = [curr_point]
for point in points_sorted[1:]:
    if point <= curr_point + eps:
        curr_cluster.append(point)
    else:
        clusters.append(curr_cluster)
        curr_cluster = [point]
    curr_point = point
clusters.append(curr_cluster)
print(clusters)
```


?- P = [1-0.1, 2-0.31,  3-0.32, 4-0.45, 5-0.35, 6-0.40, 7-0.5 ], segment_timestamps(P, 0.2, S).
P = [1-0.1, 2-0.31, 3-0.32, 4-0.45, 5-0.35, 6-0.4, 7-0.5],
S = [[1-0.1], [2-0.31, 3-0.32, 4-0.45, 5-0.35, 6-0.4, 7-0.5]] .

*/

segment_timestamps(Pairs, Epsilon, Segments) :-
    segment_timestamps(Pairs, Epsilon, [], RevRevSegments),
    reverse_segments(RevRevSegments, RevFwdIndivSegments),
    reverse(RevFwdIndivSegments, Segments).


segment_timestamps([], _Epsilon, Segments, Segments).
segment_timestamps([K-V | Pairs], Epsilon, [], SegmentsOut) :-
    segment_timestamps(Pairs, Epsilon, [[K-V]], SegmentsOut).
segment_timestamps([K-V | Pairs], Epsilon, SegmentsIn0, SegmentsOut) :-
    SegmentsIn0 = [[KLast-VLast | Vs] | Segs],
    (   Epsilon < V - VLast
    ->  SegmentsIn1 = [[K-V], [KLast-VLast | Vs] | Segs] % Create New Segment
    ;   SegmentsIn1 = [[K-V, KLast-VLast | Vs] | Segs]   % Add to last Segment
    ),
    segment_timestamps(Pairs, Epsilon, SegmentsIn1, SegmentsOut).


reverse_segments([], []).
reverse_segments([RSeg | RSegs], [Seg | Segs]) :-
    reverse(RSeg, Seg),
    reverse_segments(RSegs, Segs).


list_keyed_elems(List, IdxKeyedElems) :-
    list_keyed_elems_(List, 0, IdxKeyedElems).

list_keyed_elems_([], _N0, []).
list_keyed_elems_([Elem | List], N0, [N1-Elem | IdxKeyedElems]) :-
    N1 is N0 + 1,
    list_keyed_elems_(List, N1, IdxKeyedElems).



/* end of date_spectra.pl */














