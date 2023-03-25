#python ~/TuSeisSolScripts/TeleseismicDataRelated/compute_multi_cmt.py temporal $1 1 ../ThirdParty/muGuvercin.dat --time_range 100 132  --time_sub 4.5 13.0 22  --NZ 1 --proj "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=37 +lat_0=37"
#python ~/TuSeisSolScripts/TeleseismicDataRelated/compute_multi_cmt.py temporal $1 1 ../ThirdParty/muGuvercin.dat --time_range 0 85  --time_sub 0 13.5 31.1 42.1 50.3 66.7  --NZ 1 --proj "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=37 +lat_0=37"

python ~/TuSeisSolScripts/TeleseismicDataRelated/plot_stf_cmt.py PointSourceFile_test1_180_subshear_4_1.h5 --time_range 100 132
python ~/TuSeisSolScripts/TeleseismicDataRelated/plot_map_cmt.py PointSourceFile_test1_180_subshear_4_1.h5 --fault_edge $1 "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=37 +lat_0=37" --MapBoundaries  36 39 37 39 --scalebarSize 50.0 --ext svg --beachSize 2.0
python ~/TuSeisSolScripts/TeleseismicDataRelated/plot_stf_cmt.py PointSourceFile_test1_180_subshear_7_1.h5 --time_range 0 85
python ~/TuSeisSolScripts/TeleseismicDataRelated/plot_map_cmt.py PointSourceFile_test1_180_subshear_7_1.h5 --fault_edge $1 "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=37 +lat_0=37" --MapBoundaries  36 39 36 39 --scalebarSize 50.0 --ext svg --beachSize 2.0
