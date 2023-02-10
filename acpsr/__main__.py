import argparse
import tqdm
import os

from acpsr.model.inference import Discriminator
from acpsr.data.generate_combined_prescription import start_generation
import acpsr.data.reader as reader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'acpsr',
                    description = 'acupoint speech recognition')
    parser.add_argument('command', metavar='C', type=str, nargs='?',
                        help='command')
    args = parser.parse_args()


    if args.command == "generate":
        sample_num = start_generation()

    if args.command == "evaluation":
        model = Discriminator()
        gener_path = os.path.join('./data', 'combined_prescription')
        with open(os.path.join(gener_path, 'sample_num'), 'r') as f:
            total_samples=int(f.read())

        file_pref = os.path.join(gener_path, 'data')

        all_samples = [(file_pref+"/sample_" + str(i) + ".wav",
                        "".join(open(file_pref+"/sample_" + str(i) + ".txt"))) for i in range(total_samples)]
        all_samples = [(a, b.strip().split("\t"))for a, b in all_samples]

        correct = 0
        total_pred = 0
        total_ref = 0

        for src, ref in tqdm.tqdm(all_samples):
            pred = model.inference(src)
            correct += sum([1 for item in pred if item in ref])
            total_pred += len(pred)
            total_ref += len(ref)

        prec = correct / total_pred
        recl = correct / total_ref
        f1 = 2 * prec * recl / (prec + recl)
        print("Precision: %f" % (correct / total_pred))
        print("Recall:    %f" % (correct / total_ref))
        print("F1:        %f" % (f1))