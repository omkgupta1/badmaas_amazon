# Business entity resolution — run summary

## Timings

| stage | ok | minutes | peak RSS GB |
|---|---|---|---|
| prep[train] load | True | 0.05 | 4.34 |
| prep[train] indic lexicon | True | 0.17 | 4.34 |
| prep[train] load | True | 0.08 | 2.4 |
| prep[train] indic lexicon | True | 0.32 | 3.2 |
| prep[train] pass 1 (lite parse) | True | 2.13 | 4.34 |
| prep[train] pass 1 (lite parse) | True | 2.44 | 3.2 |
| prep[train] address lexicon | True | 0.99 | 4.34 |
| prep[train] address lexicon | True | 1.32 | 3.2 |
| prep[train] pass 2 (full parse) | True | 5.69 | 4.34 |
| prep[test] load | True | 0.2 | 1.56 |
| prep[test] indic lexicon | True | 0.0 | 1.56 |
| prep[train] pass 2 (full parse) | True | 6.24 | 3.2 |
| prep[test] load | True | 0.1 | 2.34 |
| prep[test] indic lexicon | True | 0.0 | 2.34 |
| prep[test] pass 1 (lite parse) | True | 4.13 | 1.56 |
| prep[test] address lexicon | True | 0.76 | 2.07 |
| prep[test] pass 1 (lite parse) | True | 3.51 | 2.34 |
| prep[test] address lexicon | True | 0.76 | 2.4 |
| prep[test] pass 2 (full parse) | True | 4.38 | 2.07 |
| prep[test] pass 2 (full parse) | True | 4.25 | 2.4 |
| blocking[train] retrieve | True | 829.71 | 5.47 |
| blocking[train] stage-0 ranker | True | 0.76 | 5.47 |
| blocking[train] cut to top-M | True | 1.81 | 5.47 |
| blocking[train] report | True | 0.03 | 5.47 |
| blocking[test] retrieve | True | 289.55 | 5.64 |
| blocking[test] report | True | 0.02 | 5.64 |
| features[train] setup | True | 0.89 | 6.5 |
| features[train] pairs | True | 14.3 | 8.06 |
| context[train] India | True | 1.8 | 8.34 |
| context[train] US | True | 2.51 | 8.66 |
| features[test] setup | True | 0.83 | 8.34 |
| features[test] pairs | True | 14.64 | 8.34 |
| context[test] France | True | 0.54 | 8.37 |
| context[test] India | True | 2.43 | 8.37 |
| context[test] US | True | 1.57 | 8.37 |
| train[A] fold 0 | True | 6.05 | 6.7 |
| train[A] fold 1 | True | 5.47 | 6.7 |
| train[A] fold 2 | True | 4.84 | 6.7 |
| predict[A] train OOF | True | 16.0 | 6.7 |
| decide[train] calibrate + assign | True | 0.13 | 6.7 |
| decide[train] singleton model | True | 0.29 | 7.75 |
| decide[train] tune rule | True | 2.67 | 7.75 |
| collective[train] India | True | 5.52 | 7.65 |
| collective[train] US | True | 7.5 | 8.3 |
| orphan model [train] | True | 1.47 | 8.3 |
| collective[test] France | True | 0.57 | 9.24 |
| collective[test] India | True | 6.61 | 9.24 |
| collective[test] US | True | 2.64 | 9.24 |
| orphan model [test] | True | 0.68 | 9.24 |
| train[B] fold 0 | True | 3.09 | 6.13 |
| train[B] fold 1 | True | 3.42 | 6.13 |
| train[B] fold 2 | True | 3.33 | 6.13 |
| predict[B] train OOF | True | 5.0 | 6.82 |
| decide[train] calibrate + assign | True | 0.22 | 7.34 |
| decide[train] singleton model | True | 0.53 | 7.89 |
| decide[train] tune rule | True | 4.27 | 7.89 |
| write[test] decide | True | 0.46 | 7.89 |
| write[test] tsv files | True | 2.77 | 7.89 |

## Prep coverage (train)

```json
{
 "s1": [
  {
   "country": "India",
   "src": 1,
   "n": 883188,
   "has_hn": 0.8815439068465604,
   "has_state": 1.0,
   "has_city": 0.99780001539876,
   "has_street": 0.7007454811433126,
   "addr_empty": 0.0,
   "is_domain": 0.0,
   "has_alias": 2.264523521605819e-06,
   "indic_name": 0.0
  },
  {
   "country": "US",
   "src": 1,
   "n": 1323633,
   "has_hn": 0.9999984890071493,
   "has_state": 1.0,
   "has_city": 0.9995806994839204,
   "has_street": 0.9464171715271529,
   "addr_empty": 0.0,
   "is_domain": 0.0,
   "has_alias": 0.0,
   "indic_name": 0.0
  }
 ],
 "q": [
  {
   "country": "India",
   "src": 2,
   "n": 2017799,
   "has_hn": 0.8876994190204277,
   "has_state": 0.9713321297116313,
   "has_city": 0.9596669440315908,
   "has_street": 0.66674034430585,
   "addr_empty": 0.028667870288368664,
   "is_domain": 0.03305879326929986,
   "has_alias": 0.002676678896163592,
   "indic_name": 0.23508040196273267
  },
  {
   "country": "India",
   "src": 3,
   "n": 2115547,
   "has_hn": 0.8714611398375929,
   "has_state": 0.9692996657602029,
   "has_city": 0.9647651411195308,
   "has_street": 0.5944292421770824,
   "addr_empty": 0.030700334239797084,
   "is_domain": 0.03554825300501478,
   "has_alias": 0.025073420727594328,
   "indic_name": 0.1316557845323219
  },
  {
   "country": "US",
   "src": 2,
   "n": 3016817,
   "has_hn": 0.895946290411384,
   "has_state": 0.9631661449799573,
   "has_city": 0.9628770986108869,
   "has_street": 0.9095613025251449,
   "addr_empty": 0.03683385502004265,
   "is_domain": 0.048112629967280086,
   "has_alias": 0.003382704353628344,
   "indic_name": 0.0
  },
  {
   "country": "US",
   "src": 3,
   "n": 3170056,
   "has_hn": 0.9028581829469259,
   "has_state": 0.9649949401524768,
   "has_city": 0.964171610848515,
   "has_street": 0.9115331085633819,
   "addr_empty": 0.035005059847523196,
   "is_domain": 0.04613546259119713,
   "has_alias": 0.03812424764736017,
   "indic_name": 0.0
  }
 ]
}
```

## Prep coverage (test)

```json
{
 "s1": [
  {
   "country": "France",
   "src": 1,
   "n": 259452,
   "has_hn": 0.9950549619968241,
   "has_state": 1.0,
   "has_city": 1.0,
   "has_street": 0.9978493131677536,
   "addr_empty": 0.0,
   "is_domain": 0.0,
   "has_alias": 0.0,
   "indic_name": 0.0
  },
  {
   "country": "India",
   "src": 1,
   "n": 809986,
   "has_hn": 0.8813806658386688,
   "has_state": 1.0,
   "has_city": 0.9978542839012032,
   "has_street": 0.6992676416629423,
   "addr_empty": 0.0,
   "is_domain": 0.0,
   "has_alias": 2.469178479628043e-06,
   "indic_name": 0.0
  },
  {
   "country": "US",
   "src": 1,
   "n": 663106,
   "has_hn": 0.9999954758364424,
   "has_state": 1.0,
   "has_city": 0.9996063977704922,
   "has_street": 0.9466435230566456,
   "addr_empty": 0.0,
   "is_domain": 0.0,
   "has_alias": 0.0,
   "indic_name": 0.0
  }
 ],
 "q": [
  {
   "country": "France",
   "src": 2,
   "n": 703378,
   "has_hn": 0.9264662812882973,
   "has_state": 0.6423701054056282,
   "has_city": 0.9693806175342419,
   "has_street": 0.9589509481388385,
   "addr_empty": 0.0306193824657581,
   "is_domain": 0.04220348091637782,
   "has_alias": 0.0,
   "indic_name": 0.0
  },
  {
   "country": "France",
   "src": 3,
   "n": 731615,
   "has_hn": 0.928931200153086,
   "has_state": 0.6592224052267928,
   "has_city": 0.9705569185978964,
   "has_street": 0.9602345495923402,
   "addr_empty": 0.029443081402103565,
   "is_domain": 0.041186963088509665,
   "has_alias": 0.018963525898184154,
   "indic_name": 0.0
  },
  {
   "country": "India",
   "src": 2,
   "n": 2312565,
   "has_hn": 0.9051425581551221,
   "has_state": 0.9771837764560132,
   "has_city": 0.9650245506612787,
   "has_street": 0.6796842467130654,
   "addr_empty": 0.022816223543986873,
   "is_domain": 0.02631277391121979,
   "has_alias": 0.002508037611915773,
   "indic_name": 0.2363635184308333
  },
  {
   "country": "India",
   "src": 3,
   "n": 2405000,
   "has_hn": 0.8897093555093555,
   "has_state": 0.9753679833679834,
   "has_city": 0.9705758835758835,
   "has_street": 0.6053347193347194,
   "addr_empty": 0.024632016632016633,
   "is_domain": 0.02848066528066528,
   "has_alias": 0.02042869022869023,
   "indic_name": 0.13332182952182953
  },
  {
   "country": "US",
   "src": 2,
   "n": 1871330,
   "has_hn": 0.9156717414886738,
   "has_state": 0.9705519603704317,
   "has_city": 0.9702612580357286,
   "has_street": 0.9170654026815153,
   "addr_empty": 0.029448039629568275,
   "is_domain": 0.03923199008192035,
   "has_alias": 0.0032484917144490815,
   "indic_name": 0.0
  },
  {
   "country": "US",
   "src": 3,
   "n": 1945701,
   "has_hn": 0.9209976250204939,
   "has_state": 0.9715696296604669,
   "has_city": 0.970835703944234,
   "has_street": 0.9183615570943325,
   "addr_empty": 0.028430370339533155,
   "is_domain": 0.03768204878344617,
   "has_alias": 0.03142260809857218,
   "indic_name": 0.0
  }
 ]
}
```

## Stage-0 ranker

```json
{
 "m": 12,
 "n_models": 2,
 "cross_fitted": true,
 "recall_within_union": {
  "1": 0.9840417677460184,
  "2": 0.991206096403333,
  "3": 0.9937691171817319,
  "4": 0.995158738529691,
  "5": 0.9960025313785466,
  "6": 0.9966749288049784,
  "7": 0.9970730935555321,
  "8": 0.9974501634848645,
  "9": 0.9977349435713533,
  "10": 0.9980170868051893,
  "11": 0.9982385824280139,
  "12": 0.998407340997785
 },
 "n_eval_pos": 379240,
 "best_iteration": [
  79,
  82
 ],
 "importance": {
  "cos_n": 184176.6,
  "cos_a": 99936.2,
  "cos_na": 356533.5,
  "r_n": 41322.8,
  "r_a": 11675.2,
  "r_na": 13476556.8,
  "in_n": 9902.2,
  "in_a": 2457.4,
  "in_na": 12095.4,
  "key_name": 67208.5,
  "key_addr": 13713.7,
  "rev": 29586.7,
  "hn_eq0": 642425.4,
  "name_ratio0": 193501.3,
  "addr_tset0": 480484.3,
  "n_union": 14469.2,
  "b_addr_empty0": 5161.5,
  "b_is_domain0": 32608.0
 }
}
```

## Blocking (train)

```json
{
 "pairs": 123842490,
 "cand_per_query_mean": 11.999986628190738,
 "queries_without_candidates": 8,
 "cand_per_s1_mean": 56.118049447599056,
 "reduction_ratio": 0.9999895400394573,
 "pair_recall": 0.9905126293388703,
 "oracle_macro_f05": 0.9971733519457765,
 "pair_recall_by_country_source": {
  "India/S2": 0.9904406823163092,
  "India/S3": 0.9854530303970498,
  "US/S2": 0.9922162566638079,
  "US/S3": 0.9923418312302785
 }
}
```

## Blocking (test)

```json
{
 "pairs": 119634981,
 "cand_per_query_mean": 11.999991273461724,
 "queries_without_candidates": 4,
 "cand_per_s1_mean": 69.05162639448118,
 "reduction_ratio": 0.9999822092730513
}
```

## Orphan model

```json
{
 "oof_auc": 0.9998217772654361,
 "positive_rate": 0.7401357293128967
}
```

## Decision on train OOF (stage A)

```json
{
 "tag": "A",
 "rule": {
  "kind": "E",
  "delta": 0.5,
  "item_floor": 0.0,
  "mhat": 1.0,
  "score": 0.9893058458666492
 },
 "oof_macro_f05": 0.9893058458666492,
 "breakdown": {
  "country=India": 0.9885643188526814,
  "country=US": 0.9898006263385868,
  "singletons": 0.9895332137901937,
  "has_matches": 0.9892923966613543,
  "n_true=1": 0.9659613941280888,
  "n_true=2-3": 0.9891540734149605,
  "n_true>=4": 0.9920375584971889
 },
 "pair_metrics_calibrated": {
  "auc": 0.9999658268597246,
  "ap": 0.9994780933588486,
  "logloss": 0.0032145108561962843,
  "n": 5000000
 },
 "kept_pairs": 7428333,
 "kept_precision": 0.998484585976423,
 "kept_recall_of_all_true": 0.971029271316571,
 "top_rules": [
  {
   "kind": "E",
   "delta": 0.5,
   "item_floor": 0.0,
   "mhat": 1.0,
   "score": 0.9893058458666492
  },
  {
   "kind": "E",
   "delta": 0.5,
   "item_floor": 0.1,
   "mhat": 1.0,
   "score": 0.9893058458666492
  },
  {
   "kind": "E",
   "delta": 0.5,
   "item_floor": 0.2,
   "mhat": 1.0,
   "score": 0.9893058458666492
  },
  {
   "kind": "E",
   "delta": 0.5,
   "item_floor": 0.3,
   "mhat": 1.0,
   "score": 0.9893058458666492
  },
  {
   "kind": "E",
   "delta": 0.4,
   "item_floor": 0.0,
   "mhat": 1.0,
   "score": 0.9893030337370993
  },
  {
   "kind": "E",
   "delta": 0.4,
   "item_floor": 0.1,
   "mhat": 1.0,
   "score": 0.9893030337370993
  },
  {
   "kind": "E",
   "delta": 0.4,
   "item_floor": 0.2,
   "mhat": 1.0,
   "score": 0.9893030337370993
  },
  {
   "kind": "E",
   "delta": 0.4,
   "item_floor": 0.3,
   "mhat": 1.0,
   "score": 0.9893030337370993
  },
  {
   "kind": "E0",
   "delta": 0.5,
   "item_floor": 0.0,
   "mhat": 1.0,
   "score": 0.9892808395232215
  },
  {
   "kind": "E0",
   "delta": 0.5,
   "item_floor": 0.1,
   "mhat": 1.0,
   "score": 0.9892808395232215
  },
  {
   "kind": "E0",
   "delta": 0.5,
   "item_floor": 0.2,
   "mhat": 1.0,
   "score": 0.9892808395232215
  },
  {
   "kind": "E0",
   "delta": 0.5,
   "item_floor": 0.3,
   "mhat": 1.0,
   "score": 0.9892808395232215
  },
  {
   "kind": "E0",
   "delta": 0.4,
   "item_floor": 0.0,
   "mhat": 1.0,
   "score": 0.989278719855148
  },
  {
   "kind": "E0",
   "delta": 0.4,
   "item_floor": 0.1,
   "mhat": 1.0,
   "score": 0.989278719855148
  },
  {
   "kind": "E0",
   "delta": 0.4,
   "item_floor": 0.2,
   "mhat": 1.0,
   "score": 0.989278719855148
  }
 ],
 "pair_prior_train": 0.061092901152100546,
 "assigned_p_hist": [
  2705280,
  41539,
  26767,
  41125,
  29121,
  29652,
  15709,
  6519,
  37708,
  7386791
 ]
}
```

## Decision on train OOF (stage B)

```json
{
 "tag": "B",
 "rule": {
  "kind": "E0",
  "delta": 0.4,
  "item_floor": 0.0,
  "mhat": 0.25,
  "score": 0.9901000038171148
 },
 "oof_macro_f05": 0.9901000038171148,
 "breakdown": {
  "country=India": 0.9895468016476742,
  "country=US": 0.9904691254071811,
  "singletons": 0.9912857919462543,
  "has_matches": 0.9900298624016661,
  "n_true=1": 0.9656195338703191,
  "n_true=2-3": 0.9902081704661936,
  "n_true>=4": 0.9926254769161628
 },
 "pair_metrics_calibrated": {
  "auc": 0.9999703533377414,
  "ap": 0.9995633182210651,
  "logloss": 0.002884246874600649,
  "n": 5000000
 },
 "kept_pairs": 7444929,
 "kept_precision": 0.9985218126324643,
 "kept_recall_of_all_true": 0.9732349789516473,
 "top_rules": [
  {
   "kind": "E0",
   "delta": 0.4,
   "item_floor": 0.0,
   "mhat": 0.25,
   "score": 0.9901000038171148
  },
  {
   "kind": "E0",
   "delta": 0.4,
   "item_floor": 0.1,
   "mhat": 0.25,
   "score": 0.9901000038171148
  },
  {
   "kind": "E0",
   "delta": 0.4,
   "item_floor": 0.2,
   "mhat": 0.25,
   "score": 0.9901000038171148
  },
  {
   "kind": "E0",
   "delta": 0.4,
   "item_floor": 0.3,
   "mhat": 0.25,
   "score": 0.9901000038171148
  },
  {
   "kind": "E0",
   "delta": 0.4,
   "item_floor": 0.0,
   "mhat": 0.5,
   "score": 0.9900990496458414
  },
  {
   "kind": "E0",
   "delta": 0.4,
   "item_floor": 0.1,
   "mhat": 0.5,
   "score": 0.9900990496458414
  },
  {
   "kind": "E0",
   "delta": 0.4,
   "item_floor": 0.2,
   "mhat": 0.5,
   "score": 0.9900990496458414
  },
  {
   "kind": "E0",
   "delta": 0.4,
   "item_floor": 0.3,
   "mhat": 0.5,
   "score": 0.9900990496458414
  },
  {
   "kind": "E0",
   "delta": 0.3,
   "item_floor": 0.0,
   "mhat": 0.25,
   "score": 0.990095957016077
  },
  {
   "kind": "E0",
   "delta": 0.3,
   "item_floor": 0.1,
   "mhat": 0.25,
   "score": 0.990095957016077
  },
  {
   "kind": "E0",
   "delta": 0.3,
   "item_floor": 0.2,
   "mhat": 0.25,
   "score": 0.990095957016077
  },
  {
   "kind": "E0",
   "delta": 0.3,
   "item_floor": 0.3,
   "mhat": 0.25,
   "score": 0.990095957016077
  },
  {
   "kind": "E",
   "delta": 0.5,
   "item_floor": 0.0,
   "mhat": 0.5,
   "score": 0.9900928870249102
  },
  {
   "kind": "E",
   "delta": 0.5,
   "item_floor": 0.1,
   "mhat": 0.5,
   "score": 0.9900928870249102
  },
  {
   "kind": "E",
   "delta": 0.5,
   "item_floor": 0.2,
   "mhat": 0.5,
   "score": 0.9900928870249102
  }
 ],
 "pair_prior_train": 0.061092901152100546,
 "assigned_p_hist": [
  2724776,
  20666,
  35928,
  13198,
  48969,
  13362,
  10629,
  25090,
  19790,
  7407803
 ]
}
```

## Test prediction stats

```json
{
 "France": {
  "n_s1": 259452,
  "share_s1_with_match": 0.9435194178499299,
  "mean_matches_per_s1": 3.4019047839292047,
  "n_queries": 1434993,
  "share_queries_assigned": 0.6150768679707845,
  "mean_top_p": 0.642750322133333,
  "share_top_p_ge_0.5": 0.6361891660795558
 },
 "India": {
  "n_s1": 809986,
  "share_s1_with_match": 0.9418323279661623,
  "mean_matches_per_s1": 3.3922759158800275,
  "n_queries": 4717565,
  "share_queries_assigned": 0.5824394576439328,
  "mean_top_p": 0.5986923200477339,
  "share_top_p_ge_0.5": 0.5925380996340273
 },
 "US": {
  "n_s1": 663106,
  "share_s1_with_match": 0.9433333433870301,
  "mean_matches_per_s1": 3.5488504100400236,
  "n_queries": 3817031,
  "share_queries_assigned": 0.6165168687390802,
  "mean_top_p": 0.6451078064779349,
  "share_top_p_ge_0.5": 0.6371195308605039
 },
 "pair_mean_p": 0.052289758765869505
}
```

## Outputs

```json
{
 "s1_rows": 1732544,
 "s1_with_matches": 1633199,
 "matched_pairs": 5983591,
 "s1_with_candidates": 1732544,
 "candidate_pairs": 119634981,
 "validator": {
  "ran": true,
  "exit_code": 0,
  "stdout": "ML Challenge 2026 — submission validator\n  test dir: /Users/omgupta/Downloads/student_resource/dataset/test\n  required S1 entities: 1732544\n  valid S2/S3 match IDs: 9969589\n  matching_results.tsv: 1732544 rows (99345 empty, 1633199 non-empty).\n  candidate_pairs.tsv: 1732544 rows (0 empty, 1732544 non-empty).\n\nPASS — no blocking issues found. Safe to submit.\n"
 }
}
```

## Model A: best iterations [661, 598, 494]

| feature | gain |
|---|---|
| s0 | 167686162.0 |
| q_margin_s0 | 80782391.8 |
| s0_rank | 9435339.3 |
| q_margin_cos_na | 3246530.1 |
| q_top1_is_s | 2559979.0 |
| n_extra_b_maxidf | 1888304.4 |
| hn_logdiff | 1773993.6 |
| g_hn_b | 1478719.3 |
| g_hn_b_share | 1458664.9 |
| g_addr_max | 1005053.0 |
| nums_b_only | 787578.1 |
| nums_jacc | 728055.2 |
| s_rank_s0 | 589968.7 |
| q_margin_ad_tset | 459707.2 |
| legal_conflict | 438274.5 |
| q_margin_n_idf_jacc | 378117.5 |
| g_hn_a_share | 374444.4 |
| n_cat_prefix | 362462.6 |
| nums_a_only | 323901.7 |
| n_miss_a_maxidf | 306347.1 |
| ntok_diff | 304311.2 |
| b_ntok | 299562.3 |
| n_tsort | 295333.5 |
| hid_ratio | 279934.5 |
| n_len_ratio | 276143.1 |
| q_margin_n_tset | 246141.0 |
| g_n | 232487.8 |
| n_extra_b_cnt | 231626.3 |
| n_cat_partial | 227657.0 |
| n_first_eq | 224231.3 |

## Model B: best iterations [194, 219, 207]

| feature | gain |
|---|---|
| pa | 202835173.3 |
| q_pa_margin | 70561063.6 |
| q_pa_rank | 4141125.8 |
| s0_rank | 1577461.7 |
| s0 | 1363572.2 |
| p_match | 783076.3 |
| q_pa_sum | 756139.6 |
| q_margin_s0 | 347382.2 |
| q_margin_cos_na | 275914.9 |
| q_pa_max_other | 260977.0 |
| ad_tsort | 93738.2 |
| s_pa_sum_other_src | 72607.0 |
| addr_tset0 | 68200.8 |
| q_rank_cos_na | 63809.9 |
| q_margin_n_idf_jacc | 60763.3 |
| a_name_family | 52255.6 |
| s_pa_max_other | 40243.4 |
| g_name_max | 38971.8 |
| psup_a_share | 38463.0 |
| s_ncand | 38098.2 |
| s_pa_sum_other | 34801.4 |
| n_inter_idf | 32490.5 |
| fmt_n_upper | 31726.2 |
| psup_a | 28277.3 |
| ad_ratio | 27930.0 |
| q_margin_n_tset | 27453.6 |
| s_rank_s0 | 25626.3 |
| n_extra_b_maxidf | 23900.4 |
| g_addr_max | 21406.2 |
| b_addr_empty0 | 20674.7 |
