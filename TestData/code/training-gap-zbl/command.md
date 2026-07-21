export LD_LIBRARY_PATH=/home/alberto/UniPd-GitHub/Tesi/source-compilation/QUIP/builddir/src/libAtoms:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/home/alberto/UniPd-GitHub/Tesi/source-compilation/QUIP/builddir/src/GAP:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/home/alberto/UniPd-GitHub/Tesi/source-compilation/QUIP/builddir/src/Programs:$LD_LIBRARY_PATH

------------------------------------------------------------

gap_fit at_file=../train_data.extxyz \
default_sigma={0.0002 0.1 0.0 0.0} \
gap={soap l_max=2 n_max=2 atom_sigma=0.5 zeta=4 cutoff=5.0 covariance_type=dot_product n_sparse=500 sparse_method=uniform delta=1} \
e0={Ta:0:O:0} \
energy_parameter_name=energy \
force_parameter_name=forces \
core_param_file=zbl_core.xml \
core_ip_args={IP ZBL} \
gp_file=out_potenziale_gap_500_n2_l2_zbl.xml > potenziale_quippy_500_n2_l2_zbl.log

-----------------------------------------------------------------

tail -f potenziale_quippy.log
