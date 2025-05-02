import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/libmugrid/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/python"))

import muSpectre as µ
import muGrid

import numpy as np
from NuMPI import MPI
from NuMPI.Tools import Reduction
import scipy
import netCDF4

def test_in_general():
    if MPI.COMM_WORLD.rank == 0:
        print(f'MPI.COMM_WORLD.size = {MPI.COMM_WORLD.size}')
    ### ----- Parameters ------ ###
    nb_grid_pts = [3, 4, 5]
    lengths = [1, 1, 1]
    dim = 3

    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet
    Young = 2600 # in MPa
    Poisson = 0.4
    delta_eps = np.zeros((3, 3))
    delta_eps[0, 0] = 0.02

    formulation = µ.Formulation.small_strain

    # muSpectre solver parameters
    newton_tol       = 1e-7
    cg_tol           = 1e-7 # tolerance for cg algo
    equil_tol        = 1e-7 # tolerance for equilibrium
    maxiter          = 10000
    verbose          = µ.Verbosity.Silent
    fft = 'mpi' # Parallel fft

    mask_xyz = np.ones(nb_grid_pts)
    mask_xyz[1, 1, 2] = 0


    ### ----- muSpectre calculation ----- ###
    # muSpectre cell initialization
    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient,
                  weights=weights, fft=fft, communicator=MPI.COMM_WORLD)
    mask_xyz = mask_xyz[cell.fft_engine.subdomain_slices]
    mat = µ.material.MaterialLinearElastic1_3d.make(cell, "hard", Young, Poisson)
    vac = µ.material.MaterialLinearElastic1_3d.make(cell, "vacuum", 0, 0)
    mask_p = mask_xyz.flatten(order='F')
    for pixel_id, pixel in cell.pixels.enumerate():
        if mask_p[pixel_id] == 1:
            mat.add_pixel(pixel_id)
        else:
            vac.add_pixel(pixel_id)
    cell.initialise()
    shape = (dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts)

    # Krylov solver initialization
    solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)

    # Solve muSpectre
    res = µ.solvers.newton_cg(cell, delta_eps, solver, newton_tol, equil_tol,
                              verbose, μ.solvers.IsStrainInitialised.No,
                              µ.StoreNativeStress.No)
    stress_ijqxyz = res.stress.reshape(shape, order='F')

    helper_stress_xyz = stress_ijqxyz[0, 0, 0]
    stress1_xyz = helper_stress_xyz.copy()

    # Displacement for saving
    rng = np.random.default_rng(111912573496)
    displ = rng.random((3, *cell.nb_domain_grid_pts))
    displ = displ[:, *cell.fft_engine.subdomain_slices]
    displ1 = displ.copy()

    # Total strain field
    strain1_ijqxyz = res.grad.reshape(shape, order='F').copy()

    ### ----- Saving ----- ###
    name = f'results_testing/write_netcdf_mpi_size={MPI.COMM_WORLD.size}.nc'
    comm = muGrid.Communicator(MPI.COMM_WORLD)

    # create file io object
    if os.path.exists(name):
        if comm.rank == 0:
            os.remove(name)
    MPI.COMM_WORLD.Barrier() # wait for rank 0 to delete the old netcdf file
    file_io_object = muGrid.FileIONetCDF(
        name, muGrid.FileIONetCDF.OpenMode.Write, comm)

    # Register field for saving total strain
    fields = cell.get_field_collection()
    fields.register_real_field('strain_total', components_shape=[3, 3], sub_division='quad_point')
    helper = fields.get_field('strain_total').array()
    helper[:] = strain1_ijqxyz

    # Register field for saving displacements
    fields.register_real_field('fluctuating_displacement', nb_components=3, sub_division='pixel')
    fluct_displ = fields.get_field('fluctuating_displacement')
    fluct_displ = fluct_displ.array()
    fluct_displ[:] = displ[:, None, :, :, :]

    # register global fields of the cell which you want to write
    file_io_object.register_field_collection(
        field_collection=fields,
        field_names=["strain", "stress", "fluctuating_displacement",
                     "strain_total"])

    # Write data
    file_io_object.append_frame().write(["stress"])
    file_io_object.append_frame().write(["strain"])
    file_io_object.append_frame().write(["strain_total"])
    file_io_object.append_frame().write(["fluctuating_displacement"])
    file_io_object.close()


    ### ----- Loading ----- ###
    # Loading
    nc = netCDF4.Dataset(name, 'r')
    stress_tijqxyz = nc.variables['stress'][:]
    fluct_strain = nc.variables['strain'][:]
    total_strain = nc.variables['strain_total'][:]
    displ = nc.variables['fluctuating_displacement'][:]
    nc.close()
    #print('stress:', stress_tijqxyz[:, 0, 0, 0, 0, 0, 0])
    #print('fluct_strain:', fluct_strain[:, 0, 0, 0, 0, 0, 0])
    #print('total_strain:', total_strain[:, 0, 0, 0, 0, 0, 0])
    #print('displ:', displ[:, 0, 0, 0, 0, 0])

    # Load stress
    nc = netCDF4.Dataset(name, 'r')
    stress_tijqxyz = nc.variables['stress'][:]
    nc.close()
    norm_stress = np.linalg.norm(stress_tijqxyz)
    print(f'rank {MPI.COMM_WORLD.rank}: stress_shape = {stress_tijqxyz.shape}')

    stress2_xyz = stress_tijqxyz[:, :, :, :, *cell.fft_engine.subdomain_slices]
    stress2_xyz = stress2_xyz[0, 0, 0, 0]
    diff = np.linalg.norm(stress1_xyz - stress2_xyz)
    print(f'rank {MPI.COMM_WORLD.rank}: Difference stress saved vs stress loaded: {diff}')

    # Load displacement
    nc = netCDF4.Dataset(name, 'r')
    displ = nc.variables['fluctuating_displacement'][:]
    nc.close()
    displ2 = displ[-1, :, 0, *cell.fft_engine.subdomain_slices]
    diff = np.linalg.norm(displ1 - displ2)
    print(f'rank {MPI.COMM_WORLD.rank}: Difference displ saved vs displ loaded: {diff}')

    # Load strain total
    nc = netCDF4.Dataset(name, 'r')
    strain_tijqxyz = nc.variables['strain_total'][:]
    nc.close()
    strain2 = strain_tijqxyz[2, :, :, :, *cell.fft_engine.subdomain_slices]
    diff = np.linalg.norm(strain1_ijqxyz - strain2)
    print(f'rank {MPI.COMM_WORLD.rank}: Difference strain saved vs strain loaded: {diff}')

    ### ----- Load in serial ----- ###
    if MPI.COMM_WORLD.size == 1:
        print()
        print('Loading in serial:')
        nc = netCDF4.Dataset(name, 'r')
        print('Variables saved in netcdf-file:', nc.variables.keys())
        stress_tijqxyz = nc.variables['stress'][:]
        nc.close()
        stress2_xyz = stress_tijqxyz[0, 0, 0, 0].copy()
        diff = np.linalg.norm(stress1_xyz - stress2_xyz)
        print(f'Difference stress saved (serial) vs stress loaded (serial) = {diff}')

        # Saved in parallel and loaded in serial
        name2 = f'results_testing/write_netcdf_mpi_size={2}.nc'
        if os.path.exists(name2):
            nc = netCDF4.Dataset(name, 'r')
            stress_tijqxyz = nc.variables['stress'][0]
            displ = nc.variables['fluctuating_displacement'][-1, :, 0, :, :, :]
            nc.close()
            stress3_xyz = stress_tijqxyz[0, 0, 0].copy()
            diff = np.linalg.norm(stress1_xyz - stress3_xyz)
            print(f'Difference stress saved (parallel) vs stress loaded (serial) = {diff}')
            diff = np.linalg.norm(displ1 - displ)
            print(f'Difference displ saved (parallel) vs displ loaded (serial) = {diff}')
        else:
            print(f'File {name2} does not exist.')

def test_results_of_calculation():
    if MPI.COMM_WORLD.size > 1:
        raise ValueError('test_results_of_calculation is not implemented in parallel.')

    # Parameters
    folder = 'results_testing/chiral_Nuc=1x1xNucz_mpi'
    name_basis = 'data_3D_Nucz=1.nc'
    size_list = [1, 2]

    # Data to load
    strains = []
    stresses = []
    displacements = []

    # Loading
    for size in size_list:
        name = folder + f'{size}/' + name_basis
        nc = netCDF4.Dataset(name, 'r')
        strain = nc.variables['strain_total'][2]
        stress = nc.variables['stress'][0]
        displ = nc.variables['fluctuating_displacement'][3, :, 0, :, :, :]
        nc.close()
        strains.append(strain.copy())
        stresses.append(stress.copy())
        displacements.append(displ.copy())

    # Comparison strain
    norm = np.linalg.norm(strains[0])
    diff = np.linalg.norm(strains[0] - strains[1])
    if norm < 1e-10:
        print('ERROR: The first strain does not contain values.')
    diff = diff / norm * 100
    print(f'Difference between parallel and serial strain (%) = {diff}')

    # Comparison stress
    norm = np.linalg.norm(stresses[0])
    diff = np.linalg.norm(stresses[0] - stresses[1])
    if norm < 1e-10:
        print('ERROR: The first stress does not contain values.')
    diff = diff / norm * 100
    print(f'Difference between parallel and serial stress (%) = {diff}')

    i_max = np.argmax(np.absolute(stresses[0] - stresses[1]))
    max_diff = abs(stresses[0].flatten()[i_max] - stresses[1].flatten()[i_max])
    # print(f'Maximal difference of stress entries (-): {max_diff}')

    # Comparison fluctuating displacement
    norm = np.linalg.norm(displacements[0])
    diff = np.linalg.norm(displacements[0] - displacements[1])
    if norm < 1e-10:
        print('ERROR: The first displacement does not contain values.')
    diff = diff / norm * 100
    print(f'Difference between parallel and serial displacements (%) = {diff}')


if __name__ == "__main__":
    # test_in_general()
    test_results_of_calculation()
