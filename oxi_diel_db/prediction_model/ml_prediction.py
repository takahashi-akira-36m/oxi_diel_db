from copy import deepcopy
import os
from typing import Union, List, Callable

import numpy as np
import joblib

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition

from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.composition import (
    Stoichiometry, ElementProperty, ValenceOrbital, IonProperty, TMetalFraction, ElectronegativityDiff,
    BandCenter, OxidationStates, AtomicOrbitals, AtomicPackingEfficiency)
from matminer.featurizers.structure import DensityFeatures, MinimumRelativeDistances, StructuralHeterogeneity
from matminer.featurizers.site import (
    OPSiteFingerprint, EwaldSiteEnergy, VoronoiFingerprint, LocalPropertyDifference, GaussianSymmFunc)


__author__ = "Akira Takahashi, Jun Miyamoto"


class ScalarFeaturizer:
    def __init__(self, featurizer: BaseFeaturizer, composition: Composition):
        self._featurizer = featurizer
        self._composition = composition
        self._values = featurizer.featurize(self._composition)

    @property
    def labels(self) -> List[str]:
        return self._featurizer.feature_labels()

    def get_from_label(self, label: str) -> float:
        return self._values[self.labels.index(label)]


class SiteFeaturizer:
    def __init__(self, featurizer: BaseFeaturizer, structure: Structure):
        self._featurizer = featurizer
        self._structure = structure
        self._vectors = np.transpose([self._featurizer.featurize(structure, i) for i in range(structure.num_sites)])

    @property
    def labels(self) -> List[str]:
        return self._featurizer.feature_labels()

    def get_from_label_func(self, label: str, func: Callable) -> float:
        return func(self._vectors[self.labels.index(label)])


def predict_log10_eps(target: Union[Structure, Composition],
                      dielectric_type: str,
                      model_type: str,
                      ) -> float:
    """
    :param target: structure or composition to predict dielectric constants
    :param dielectric_type: "el" or "ion"
    :param model_type: "comp" or "comp_st"
    :return: Descriptor vector
    """
    if dielectric_type not in ["el", "ion"]:
        raise ValueError(f'Specify dielectric type "el" or "ion"\nInput: {dielectric_type}')
    if model_type not in ["comp", "comp_st"]:
        raise ValueError(f'Specify regression_type "comp" or "comp_st"\nInput: {model_type}')

    if model_type == "comp":
        if isinstance(target, Structure):
            comp = target.composition
        else:
            comp = target
        comp_oxi = comp.add_charges_from_oxi_state_guesses()
        if dielectric_type == "el":
            ep = ScalarFeaturizer(ElementProperty.from_preset("matminer"), comp)
            valence = ScalarFeaturizer(ValenceOrbital(), comp)
            ion_prop = ScalarFeaturizer(IonProperty(), comp)
            en_diff = ScalarFeaturizer(ElectronegativityDiff(), comp_oxi)
            oxi_state = ScalarFeaturizer(OxidationStates.from_preset("deml"), comp_oxi)
            atomic_orbital = ScalarFeaturizer(AtomicOrbitals(), comp)
            descriptor = [
                ep.get_from_label("PymatgenData minimum X"),
                ep.get_from_label("PymatgenData range X"),
                ep.get_from_label("PymatgenData std_dev X"),
                ep.get_from_label("PymatgenData mean row"),
                ep.get_from_label("PymatgenData std_dev row"),
                ep.get_from_label("PymatgenData mean group"),
                ep.get_from_label("PymatgenData mean block"),
                ep.get_from_label("PymatgenData std_dev block"),
                ep.get_from_label("PymatgenData mean atomic_mass"),
                ep.get_from_label("PymatgenData std_dev atomic_mass"),
                ep.get_from_label("PymatgenData std_dev atomic_radius"),
                ep.get_from_label("PymatgenData minimum mendeleev_no"),
                ep.get_from_label("PymatgenData range mendeleev_no"),
                ep.get_from_label("PymatgenData std_dev mendeleev_no"),
                ep.get_from_label("PymatgenData mean thermal_conductivity"),
                ep.get_from_label("PymatgenData std_dev thermal_conductivity"),
                ep.get_from_label("PymatgenData mean melting_point"),
                ep.get_from_label("PymatgenData std_dev melting_point"),
                valence.get_from_label("avg s valence electrons"),
                valence.get_from_label("avg d valence electrons"),
                valence.get_from_label("frac s valence electrons"),
                valence.get_from_label("frac p valence electrons"),
                valence.get_from_label("frac d valence electrons"),
                ion_prop.get_from_label("avg ionic char"),
                TMetalFraction().featurize(comp)[0],
                en_diff.get_from_label("maximum EN difference"),
                en_diff.get_from_label("range EN difference"),
                en_diff.get_from_label("mean EN difference"),
                en_diff.get_from_label("std_dev EN difference"),
                BandCenter().featurize(comp)[0],
                oxi_state.get_from_label("std_dev oxidation state"),
                atomic_orbital.get_from_label("HOMO_energy"),
                atomic_orbital.get_from_label("LUMO_energy"),
                atomic_orbital.get_from_label("gap_AO"),
            ]
        elif dielectric_type == "ion":
            stoich = ScalarFeaturizer(Stoichiometry(), comp)
            ep = ScalarFeaturizer(ElementProperty.from_preset("matminer"), comp)
            valence = ScalarFeaturizer(ValenceOrbital(), comp)
            ion_prop = ScalarFeaturizer(IonProperty(), comp)
            en_diff = ScalarFeaturizer(ElectronegativityDiff(), comp_oxi)
            oxi_state = ScalarFeaturizer(OxidationStates.from_preset("deml"), comp_oxi)
            atomic_orbital = ScalarFeaturizer(AtomicOrbitals(), comp)
            at_pack_eff = ScalarFeaturizer(AtomicPackingEfficiency(), comp)
            descriptor = [
                stoich.get_from_label("3-norm"),
                stoich.get_from_label("5-norm"),
                ep.get_from_label("PymatgenData mean X"),
                ep.get_from_label("PymatgenData mean row"),
                ep.get_from_label("PymatgenData std_dev row"),
                ep.get_from_label("PymatgenData std_dev group"),
                ep.get_from_label("PymatgenData mean block"),
                ep.get_from_label("PymatgenData std_dev block"),
                ep.get_from_label("PymatgenData maximum atomic_mass"),
                ep.get_from_label("PymatgenData range atomic_mass"),
                ep.get_from_label("PymatgenData mean atomic_mass"),
                ep.get_from_label("PymatgenData std_dev atomic_mass"),
                ep.get_from_label("PymatgenData maximum atomic_radius"),
                ep.get_from_label("PymatgenData range atomic_radius"),
                ep.get_from_label("PymatgenData mean atomic_radius"),
                ep.get_from_label("PymatgenData std_dev atomic_radius"),
                ep.get_from_label("PymatgenData minimum mendeleev_no"),
                ep.get_from_label("PymatgenData mean mendeleev_no"),
                ep.get_from_label("PymatgenData std_dev mendeleev_no"),
                ep.get_from_label("PymatgenData mean thermal_conductivity"),
                ep.get_from_label("PymatgenData std_dev thermal_conductivity"),
                ep.get_from_label("PymatgenData mean melting_point"),
                ep.get_from_label("PymatgenData std_dev melting_point"),
                valence.get_from_label("avg s valence electrons"),
                valence.get_from_label("frac s valence electrons"),
                valence.get_from_label("frac p valence electrons"),
                valence.get_from_label("frac d valence electrons"),
                ion_prop.get_from_label("avg ionic char"),
                TMetalFraction().featurize(comp)[0],
                en_diff.get_from_label("minimum EN difference"),
                en_diff.get_from_label("range EN difference"),
                en_diff.get_from_label("mean EN difference"),
                en_diff.get_from_label("std_dev EN difference"),
                oxi_state.get_from_label("range oxidation state"),
                oxi_state.get_from_label("std_dev oxidation state"),
                atomic_orbital.get_from_label("LUMO_energy"),
                atomic_orbital.get_from_label("gap_AO"),
                at_pack_eff.get_from_label("mean simul. packing efficiency"),
                at_pack_eff.get_from_label("mean abs simul. packing efficiency"),
                at_pack_eff.get_from_label("dist from 1 clusters |APE| < 0.010"),
                at_pack_eff.get_from_label("dist from 3 clusters |APE| < 0.010"),
                at_pack_eff.get_from_label("dist from 5 clusters |APE| < 0.010"),
            ]
    elif model_type == "comp_st":
        if isinstance(target, Composition):
            raise ValueError('comp_st (Using compositional and structural descriptor) is specified, '
                             'but target is composition')
        comp: Composition = target.composition
        comp_oxi = comp.add_charges_from_oxi_state_guesses()
        target_orig = deepcopy(target)
        target.add_oxidation_state_by_guess()
        if dielectric_type == "el":
            ep = ScalarFeaturizer(ElementProperty.from_preset("matminer"), comp)
            valence = ScalarFeaturizer(ValenceOrbital(), comp)
            en_diff = ScalarFeaturizer(ElectronegativityDiff(), comp_oxi)
            atomic_orbital = ScalarFeaturizer(AtomicOrbitals(), comp)
            density = ScalarFeaturizer(DensityFeatures(), target)
            dist_btw_nn = MinimumRelativeDistances().featurize(target_orig)
            opsf = SiteFeaturizer(OPSiteFingerprint(), target)
            voro_fp = SiteFeaturizer(VoronoiFingerprint(use_symm_weights=True), target)
            gsf = SiteFeaturizer(GaussianSymmFunc(), target)
            lpd = SiteFeaturizer(LocalPropertyDifference.from_preset("ward-prb-2017"), target)
            descriptor = [
                ep.get_from_label("PymatgenData std_dev X"),
                ep.get_from_label("PymatgenData mean block"),
                ep.get_from_label("PymatgenData std_dev atomic_mass"),
                valence.get_from_label("frac d valence electrons"),
                TMetalFraction().featurize(comp)[0],
                en_diff.get_from_label("maximum EN difference"),
                en_diff.get_from_label("mean EN difference"),
                atomic_orbital.get_from_label("HOMO_energy"),
                atomic_orbital.get_from_label("LUMO_energy"),
                density.get_from_label("density"),
                np.mean(dist_btw_nn),
                np.std(dist_btw_nn),
                opsf.get_from_label_func("tetrahedral CN_4", np.max),
                opsf.get_from_label_func("rectangular see-saw-like CN_4", np.max),
                np.max([EwaldSiteEnergy(accuracy=4).featurize(target, i) for i in range(target.num_sites)]),
                voro_fp.get_from_label_func("Voro_area_std_dev", np.max),
                voro_fp.get_from_label_func("Voro_area_std_dev", np.mean),
                voro_fp.get_from_label_func("Voro_dist_minimum", np.min),
                voro_fp.get_from_label_func("Voro_dist_minimum", np.std),
                gsf.get_from_label_func("G2_20.0", np.std),
                gsf.get_from_label_func("G2_80.0", np.max),
                gsf.get_from_label_func("G4_0.005_4.0_-1.0", np.mean),
                lpd.get_from_label_func("local difference in NdValence", np.mean),
                lpd.get_from_label_func("local difference in NValence", np.min),
                lpd.get_from_label_func("local difference in NValence", np.std),
                lpd.get_from_label_func("local difference in NdUnfilled", np.mean),
                lpd.get_from_label_func("local difference in NUnfilled", np.min),
                lpd.get_from_label_func("local difference in NUnfilled", np.mean),
                lpd.get_from_label_func("local difference in GSmagmom", np.mean)
            ]
        elif dielectric_type == "ion":
            ep = ScalarFeaturizer(ElementProperty.from_preset("matminer"), comp)
            atomic_orbitals = ScalarFeaturizer(AtomicOrbitals(), comp)
            density = ScalarFeaturizer(DensityFeatures(), target)
            str_het = ScalarFeaturizer(StructuralHeterogeneity(), target)
            opsf = SiteFeaturizer(OPSiteFingerprint(), target)
            voro_fp = SiteFeaturizer(VoronoiFingerprint(use_symm_weights=True), target)
            gsf = SiteFeaturizer(GaussianSymmFunc(), target)
            lpd = SiteFeaturizer(LocalPropertyDifference.from_preset("ward-prb-2017"), target)
            descriptor = [
                ep.get_from_label("PymatgenData std_dev row"),
                ep.get_from_label("PymatgenData mean thermal_conductivity"),
                ep.get_from_label("PymatgenData std_dev melting_point"),
                TMetalFraction().featurize(comp)[0],
                atomic_orbitals.get_from_label("gap_AO"),
                density.get_from_label("density"),
                density.get_from_label("packing fraction"),
                str_het.get_from_label("mean neighbor distance variation"),
                str_het.get_from_label("avg_dev neighbor distance variation"),
                opsf.get_from_label_func("sgl_bd CN_1", np.mean),
                opsf.get_from_label_func("bent 150 degrees CN_2", np.mean),
                opsf.get_from_label_func("linear CN_2", np.mean),
                opsf.get_from_label_func("trigonal planar CN_3", np.mean),
                opsf.get_from_label_func("pentagonal planar CN_5", np.std),
                opsf.get_from_label_func("octahedral CN_6", np.max),
                opsf.get_from_label_func("octahedral CN_6", np.std),
                opsf.get_from_label_func("q6 CN_12", np.mean),
                np.max([EwaldSiteEnergy(accuracy=4).featurize(target, i) for i in range(target.num_sites)]),
                voro_fp.get_from_label_func("Symmetry_weighted_index_4", np.std),
                voro_fp.get_from_label_func("Voro_vol_maximum", np.mean),
                voro_fp.get_from_label_func("Voro_area_std_dev", np.mean),
                voro_fp.get_from_label_func("Voro_area_minimum", np.std),
                voro_fp.get_from_label_func("Voro_area_maximum", np.min),
                voro_fp.get_from_label_func("Voro_dist_std_dev", np.mean),
                gsf.get_from_label_func("G2_80.0", np.min),
                gsf.get_from_label_func("G4_0.005_4.0_1.0", np.std),
                lpd.get_from_label_func("local difference in Number", np.max),
                lpd.get_from_label_func("local difference in MendeleevNumber", np.max),
                lpd.get_from_label_func("local difference in MendeleevNumber", np.min),
                lpd.get_from_label_func("local difference in AtomicWeight", np.max),
                lpd.get_from_label_func("local difference in AtomicWeight", np.mean),
                lpd.get_from_label_func("local difference in MeltingT", np.mean),
                lpd.get_from_label_func("local difference in Row", np.max),
                lpd.get_from_label_func("local difference in Electronegativity", np.min),
                lpd.get_from_label_func("local difference in NValence", np.std),
                lpd.get_from_label_func("local difference in NsUnfilled", np.mean),
                lpd.get_from_label_func("local difference in NdUnfilled", np.max),
                lpd.get_from_label_func("local difference in NdUnfilled", np.std),
                lpd.get_from_label_func("local difference in NUnfilled", np.max),
                lpd.get_from_label_func("local difference in NUnfilled", np.min),
                lpd.get_from_label_func("local difference in NUnfilled", np.mean),
                lpd.get_from_label_func("local difference in NUnfilled", np.std),
                lpd.get_from_label_func("local difference in GSvolume_pa", np.max),
                lpd.get_from_label_func("local difference in GSvolume_pa", np.min),
                lpd.get_from_label_func("local difference in SpaceGroupNumber", np.max),
            ]
    with open(f"{os.path.dirname(__file__)}/{dielectric_type}_{model_type}.joblib", "rb") as fr:
        model: RandomForestRegressor = joblib.load(fr)
    with open(f"{os.path.dirname(__file__)}/{dielectric_type}_{model_type}_scaler.joblib", "rb") as fr:
        scaler: StandardScaler = joblib.load(fr)
    descriptor = scaler.transform([descriptor])
    return model.predict(descriptor)[0]
