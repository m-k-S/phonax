from ase.build import make_supercell
from phonax.trained_models import MACE_uniIAP_PBEsol_finetuned_model
import os
import jax

from phonax.phonons import (
    predict_hessian_matrix,
    plot_bands,
    atoms_to_ext_graph,
    to_f32,
)

import phonopy
from ase import Atoms

from compare_dispersions import compare_all_segments

import warnings
warnings.filterwarnings("ignore")

###############################
# LOAD ATOMS IN PHONOPY AND ASE
###############################

chemical = "SbO2"
phonon = phonopy.load("phonon_data/{}.yaml".format(chemical))
atoms = Atoms(
        symbols=phonon.unitcell.symbols,
        scaled_positions=phonon.unitcell.scaled_positions,
        cell=phonon.unitcell.cell,
        pbc=True
)

###############################
# PHONOPY CALCULATION WITH MACE
###############################

from janus_core.calculations.single_point import SinglePoint
from janus_core.calculations.phonons import Phonons

from phonopy.phonon.band_structure import (
    get_band_qpoints_and_path_connections,
    get_band_qpoints_by_seekpath,
)

sp_mace = SinglePoint(
    struct=atoms.copy(),
    arch="mace_mp",
    device='cpu',
    calc_kwargs={'model_paths':'trained-models/mace_agnesi_medium.model','default_dtype':'float64'},
)

phonons_mace = Phonons(
    struct=sp_mace.struct,
    supercell=[2, 2, 2],
    displacement=0.01,
    temp_step=10.0,
    temp_min=0.0,
    temp_max=1000.0,
    minimize=False,
    force_consts_to_hdf5=True,
    plot_to_file=True,
    symmetrize=False,
    write_full=True,
    minimize_kwargs={"filter_func": None},
    file_prefix="results/{}/".format(chemical),
    write_results=True,
)

phonons_mace.calc_bands()

q_points, labels, connections = get_band_qpoints_by_seekpath(
    phonons_mace.results["phonon"].primitive, phonons_mace.n_qpoints
)

bs_mace = phonons_mace.results['phonon'].get_band_structure_dict()

##############################################
# PHONOPY CALCULATION FROM DFT FORCE CONSTANTS
##############################################

phonon.run_band_structure(q_points, path_connections=connections, labels=labels)

bs = phonon.get_band_structure_dict()

########################################
# PHONAX CALCULATION USING SAME Q-POINTS
########################################

model_fn, params, num_message_passing, r_max = MACE_uniIAP_PBEsol_finetuned_model(os.path.join(os.getcwd(), 'trained-models'))

graph = atoms_to_ext_graph(atoms, r_max, num_message_passing)   
graph = jax.tree_util.tree_map(to_f32, graph)

H = predict_hessian_matrix(params,model_fn,graph)

p = plot_bands(atoms, graph, H, npoints=1000)
p.savefig('results/SbO2.png')

# TODO: Save figures
# TODO: Save q-points & frequencies 
# TODO: Save comparison metrics frequency RMSE
# TODO: Save time taken for completion on each 