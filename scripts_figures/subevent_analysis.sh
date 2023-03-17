python ~/TuSeisSolScripts/TeleseismicDataRelated/compute_multi_cmt.py temporal $1 1 ../ThirdParty/muGuvercin.dat --time_range 120 152  --NH 4 --NZ 1 --proj "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=37 +lat_0=37"
python ~/TuSeisSolScripts/TeleseismicDataRelated/plot_stf_cmt.py PointSourceFile_Turkey_2events_31mio_o5_4_1.h5 --time_range 120 152
python ~/TuSeisSolScripts/TeleseismicDataRelated/plot_map_cmt.py PointSourceFile_Turkey_2events_31mio_o5_4_1.h5 --fault_edge $1 "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=37 +lat_0=37" --MapBoundaries  36 39 37 39 
