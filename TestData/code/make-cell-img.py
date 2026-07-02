import ase.io
from ase.io import read, write
from ase.visualize import view

import matplotlib.pyplot as plt
from ase.io import read
from ase.visualize.plot import plot_atoms
from ase.data import colors, covalent_radii, atomic_numbers
from matplotlib.lines import Line2D

#---------------------------------------------------------------------------------------------------

# 1. Load your atom structure
# Changed to load only the first frame (index=0) to avoid MemoryError
traiettoria_caricata = ase.io.read('traiettoria_unita.traj', index=0)

#---------------------------------------------------------------------------------------------------

# 2. Generate and save the picture
# ASE automatically detects the format from the file extension
write('atomic_structure.png', traiettoria_caricata,
      rotation='30x,45y,0z',
      show_unit_cell=2,
      radii=0.9)

#-----------------------------------------------------------------------------------------------------

# 2. Create the plot layout
fig, ax = plt.subplots(figsize=(6, 6))
plot_atoms(traiettoria_caricata, ax, radii=0.7, rotation='50x,35y,0z', show_unit_cell=2)

# --- THE AUTOMATIC LEGEND ENGINE ---
# Automatically find all unique chemical elements in your specific file
unique_elements = sorted(list(set(traiettoria_caricata.get_chemical_symbols())))

legend_elements = [
    Line2D([0], [0],
           marker='o',
           color='w',
           markerfacecolor=colors.jmol_colors[atomic_numbers[el]], # Changed to jmol_colors for consistency
           markersize=12,
           label=el)
    for el in unique_elements
]
# Place the legend neatly inside the upper right corner of the plot
ax.legend(handles=legend_elements, loc='upper right', title="Elements")
# ------------------------------------------------------------------------
# Pad borders so nothing cuts off, hide raw gridlines, and save
ax.set_xlim(ax.get_xlim()[0] - 2, ax.get_xlim()[1] + 2)
ax.set_ylim(ax.get_ylim()[0] - 2, ax.get_ylim()[1] + 2)
ax.axis('on')
ax.grid('on')
# Remove numbers from the axes
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.savefig('structure_with_legend.png', dpi=200, bbox_inches='tight', transparent=True)
