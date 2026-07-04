
pip install meson

pip install ninja

conda install -c conda-forge gfortran make lapack openblas cmake pkg-config -y

git clone --recursive https://github.com/libAtoms/QUIP.git

cd QUIP/

meson setup builddir

meson setup builddir -Dgap=true -Dmpi=false

meson compile -C builddir
