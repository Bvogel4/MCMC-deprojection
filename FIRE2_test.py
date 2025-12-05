import h5py
import numpy as np
import pynbody
simpath = '/data/bk639/FIRE2/m10q/snapdir_600/snapshot_600'
s = pynbody.load(simpath)
s.set_units_system(distance=pynbody.units.Unit('1.0 kpc a h^-1'),  # length
    mass=pynbody.units.Unit('1.0e10 Msol h^-1'),  # mass
    velocity=pynbody.units.Unit('1.0 km s^-1')  # velocity
                    )
s.physical_units()
halopath = '/data/bk639/FIRE2/m10q/snapdir_600/halo_600.hdf5'

import h5py
import numpy as np
from pynbody.array import SimArray as SimArray


class RockstarHDF5Catalog:
    def __init__(self, hdf5_file):
        with h5py.File(hdf5_file, 'r') as f:
            # Read properties
            self.id = f['id'][:]
            self.mass = f['mass'][:] # Msun/h
            self.position = f['position'][:]  # kpc (physical)
            self.velocity = f['velocity'][:]  # km/s (physical)
            self.radius = f['radius'][:]  # kpc (physical)
            self.vel_circ_max = f['vel.circ.max'][:]  # km/s

            # Read various mass definitions
            self.mass_200c = f['mass.200c'][:]
            self.mass_vir = f['mass.vir'][:]

            # Cosmology
            self.h = f['cosmology:hubble'][()]
            self.box_length_kpc = f['info:box.length'][()]  # kpc (physical)

            #convert to SimArray
            self.mass = SimArray(self.mass, units='Msol')
            self.mass_200c = SimArray(self.mass_200c, units='Msol')
            self.mass_vir = SimArray(self.mass_vir, units='Msol')
            self.position = SimArray(self.position, units='kpc')
            self.velocity = SimArray(self.velocity, units='km s^-1')
            self.radius = SimArray(self.radius, units='kpc')


    def __len__(self):
        return len(self.id)

    def get_position_Mpc(self, index):
        """Get position in Mpc (physical)"""
        return self.position[index] / 1000.0


    def select_sphere(self, sim, halo_index, radius_kpc=None,
                                   radius_factor=1.0):
        """
        Select particles within a sphere around a halo.

        Parameters:
        -----------
        sim : pynbody snapshot (assumes positions in Mpc/h comoving)
        halo_index : index of halo
        radius_kpc : radius in kpc (physical). If None, uses halo radius
        radius_factor : multiply halo radius by this factor (if radius_kpc not given)
        """
        # Get halo center in kpc (physical)
        center_kpc = self.position[halo_index]
        # Get radius
        if radius_kpc is None:
            radius_kpc = self.radius[halo_index] * radius_factor
        print(f"Selecting particles within {radius_kpc:.2f} kpc of halo {halo_index} at {center_kpc} kpc")
        # Center and select
        sphere_filter = pynbody.filt.Sphere(radius_kpc, center_kpc)

        return sim[sphere_filter]

# Load catalog
halos = RockstarHDF5Catalog(halopath)

print(f"Number of halos: {len(halos)}")
print(f"h = {halos.h:.3f}")
print(f"Box size: {halos.box_length_kpc:.1f} kpc = {halos.box_length_kpc/1000:.3f} Mpc")

# Most massive halo
idx = np.argmax(halos.mass)
print(f"\nMost massive halo (index {idx}):")
print(f"  ID: {halos.id[idx]}")
print(f"  Mass: {halos.mass[idx]:.2e} Msun")
print(f"  Position: {halos.position[idx]} kpc (physical)")
print(f"  Radius: {halos.radius[idx]:.2f} kpc (physical)")
print(f"  Vmax: {halos.vel_circ_max[idx]:.1f} km/s")


halo_particles = halos.select_sphere(s, idx, radius_factor=1/10)
print(f"\nParticles in sphere around most massive halo: {len(halo_particles)}")
