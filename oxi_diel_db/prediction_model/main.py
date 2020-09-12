import argparse
import json
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from oxi_diel_db.prediction_model.ml_prediction import predict_log10_eps


def main():
    parser = argparse.ArgumentParser(description="Prediction of dielectric constant by ML models")
    parser.add_argument("-diel", dest="dielectric_type", type=str,
                        help='Predict electronic or ionic contribution to dielectric constant.'
                             'Specify "electronic" or "ionic".')
    parser.add_argument("-des", dest='descriptor_type', type=str,
                        help='ML model with only compositional descriptors or '
                             'both compositional and structural descriptors. '
                             'Specify "comp" or "comp_st".')
    parser.add_argument("-c", dest='composition', type=str,
                        help='composition (e.g., SiO2)')
    parser.add_argument("-s", dest='structure', type=str,
                        help='File name of structure (e.g., POSCAR).'
                             'If json files of this database (e.g., oxi_diel_db/data/mp-216.json) is specified,'
                             'structure is read from the json file.')
    args = parser.parse_args()

    if args.structure:
        if args.structure.endswith(".json"):
            with open(args.structure) as fr:
                d = json.load(fr)
            target = Structure.from_dict(d["structure"])
        else:
            target = Structure.from_file(args.structure)
    elif args.composition:
        target = Composition(args.composition)
    model_type = args.descriptor_type
    pred = predict_log10_eps(target, args.dielectric_type, model_type)
    print(f"Prediction result (log10): {pred}")
    print(f"Prediction result (not log value): {10**pred}")


if __name__ == "__main__":
    main()
