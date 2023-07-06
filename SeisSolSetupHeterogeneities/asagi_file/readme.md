
- *prefix_stress_change*.nc:* generated with:
```
#!/bin/bash     
# first we extract the last time step of the volume output, e.g. with 
python SeisSol/postprocessing/visualization/tools/extractDataFromUnstructuredOutput.py output_el_Turkey_31_ev1/Turkey_31M_o5_el_ev1_2500_s2_05_al065.xdmf --Data sigma_xx sigma_yy sigma_zz sigma_xy sigma_xz sigma_yz --last
# then we generate the netcdf files:
stress_file=Turkey_31M_o5_el_ev1_2500_s2_05_al065_resampled
python ~/TuSeisSolScripts/onHdf5/interpolate_seissol_data_to_grid.py --box "2.5e3 -100e3 170e3 -90e3 175e3 -30e3 2600" ${stress_file}.xdmf --Data sigma_xx sigma_yy sigma_zz sigma_xy sigma_xz sigma_yz
mv gridded_asagi_0.nc ${stress_file}_stress_change_coarse.nc             
python ~/TuSeisSolScripts/onHdf5/interpolate_seissol_data_to_grid.py --box "5e2 -80e3 150e3 50e3 150e3 -30e3 2600" ${stress_file}.xdmf --Data sigma_xx sigma_yy sigma_zz sigma_xy sigma_xz sigma_yz
mv gridded_asagi_0.nc ${stress_file}_stress_change_fine.nc
```

