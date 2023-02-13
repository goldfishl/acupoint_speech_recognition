
import tqdm
import os
import torch

from acpsr.model.inference import Discriminator
from acpsr.train.dataset import split_dataset, embedding, Acupuncture_Prescription

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def editing_distance(a, b):
    # Declaring array 'D' with rows = len(a) + 1 and columns = len(b) + 1:
    D = [[0 for i in range(len(b) + 1)] for j in range(len(a) + 1)]

    # Initialising first row:
    for i in range(len(a) + 1):
        D[i][0] = i

    # Initialising first column:
    for j in range(len(b) + 1):
        D[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                D[i][j] = D[i - 1][j - 1]
            else:
                # Adding 1 to account for the cost of operation
                insertion = 1 + D[i][j - 1]
                deletion = 1 + D[i - 1][j]
                replacement = 1 + D[i - 1][j - 1]

                # Choosing the best option:
                D[i][j] = min(insertion, deletion, replacement)
    return D[-1][-1]


def start_evaluate(seg_method):
    # Load the model
    model = Discriminator()
    gener_path = os.path.join('./data', 'combined_prescription')

    if seg_method == 'rnn_seg':
        total_dis = 0
        total_acc = 0
        path = os.path.join('./data', 'combined_prescription')
        test_dataset = Acupuncture_Prescription(path, "testing")
        for audio, trg in tqdm.tqdm(test_dataset):
            embed = embedding(audio).to(device)
            out = model.rnn_seg.model(embed)
            out = torch.argmax(out, dim=1).tolist()
            trg = torch.tensor([int(char) for char in trg]).tolist()
            dis = editing_distance(out, trg)
            total_dis += dis
            if len(trg) != len(out):
                raise("error")
            acc = (len(trg) - dis) / len(trg)
            total_acc += acc
        print("The average editing distance is", total_dis / len(test_dataset))
        print("The average accuracy of editing distance is", total_acc / len(test_dataset))
    


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
        if seg_method == "rule_seg":
            pred = model.inference(src, 'rule')
        if seg_method == "rnn_seg":
            pred = model.inference(src, 'rnn')
        correct += sum([1 for item in pred if item in ref])
        total_pred += len(pred)
        total_ref += len(ref)

    prec = correct / total_pred
    recl = correct / total_ref
    f1 = 2 * prec * recl / (prec + recl)
    print("Precision: %f" % (correct / total_pred))
    print("Recall:    %f" % (correct / total_ref))
    print("F1:        %f" % (f1))