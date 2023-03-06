# Evaluation (Sequential)
import tqdm
from acpsr.model.inference import Discriminator

def rule_based_evalation():
    model = Discriminator()

    #total_samples = 2176
    # total_samples = 100

    file_pref = "data/combined_prescription/evaluation/"

    all_samples = [(file_pref+"sample_" + str(i) + ".wav",
                    "".join(open(file_pref+"sample_" + str(i) + ".txt"))) for i in range(total_samples)]
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