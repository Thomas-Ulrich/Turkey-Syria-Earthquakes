#!/bin/bash

createfigureS4Col.sh ~/trash/output_model_2603_s205_second_event_noDCFS/Turkey_2events_31mio_o5_2603_s205_last col1
createfigureS4Col.sh ~/trash/output_model_2603_s205_second_event_noDCFS/Turkey_hom_dip70_3_o5_ev2_only_last col2
createfigureS4ColDiff.sh ~/trash/diff_Turkey_2events_31mio_o5_2603_s205_last col3
#example of use of moment_rate.py
#python moment_rate.py ~/trash/output_model_2603_s205_second_event_noDCFS/Turkey_hom_dip70_3_o5_ev2_only ~/trash/output_model_2603_s205_second_event_noDCFS/Turkey_2events_31mio_o5_2603_s205 --labels "without DCFS" "with DCFS" --t0_2nd 0. 150.
