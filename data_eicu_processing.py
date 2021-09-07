import pandas as pd
import pickle
import argparse
import os
from datasets import LabelsForData


def processing_for_dx(pats_file, dx_file, single_dx_file, multi_dx_file, out_file):
    dxes = pd.read_csv(dx_file, header=0)
    pats = pd.read_csv(pats_file, header=0)
    label4data = LabelsForData(multi_dx_file, single_dx_file)

    # unique patient:count
    pat_vc = pats.uniquepid.value_counts()

    # patients whose admission number is at least 2
    pat_two_plus = pat_vc[pat_vc > 1].index.tolist()

    # pid mapping admission list
    print('pid mapping admission list')
    pid_adm_map = {}
    for pid in pat_two_plus:
        pats_adm = pats[pats.uniquepid == pid]
        sorted_adms = pats_adm.sort_values(by=['hospitaldischargeyear', 'hospitaladmitoffset'],
                                           ascending=[True, False])['patientunitstayid'].tolist()
        pid_adm_map[pid] = sorted_adms

    # filter null in icc9code field
    dxes = dxes[dxes.icd9code.notnull()]

    # Building Building strSeqs
    print('Building Building strSeqs')
    seqs = []
    for pid, adms in pid_adm_map.items():
        seq = []
        for adm in adms:
            code_list = []
            diags = dxes[dxes.patientunitstayid == adm]
            for index, row in diags.iterrows():
                codes = row.icd9code.split(',')
                if len(codes) == 2:
                    # if the first letter is digit, it is icd9 code
                    if codes[0][0].isdigit():
                        code_list.append(codes[0].replace('.', ''))
                    if codes[1][0].isdigit():
                        code_list.append(codes[0].replace('.', ''))
                else:
                    if codes[0][0].isdigit():
                        code_list.append(codes[0].replace('.', ''))
            if len(code_list) > 0:
                seq.append(code_list)
        if len(seq) > 1:
            seqs.append(seq)

    # Building Building new strSeqs, which filters the admission with only one diagnosis code
    print('Building Building new strSeqs, which filters the admission with only one diagnosis code')
    new_seqs = []
    for seq in seqs:
        new_seq = []
        for adm in seq:
            if len(adm) == 1:
                continue
            else:
                code_set = set(adm)
                if len(code_set) == 1:
                    continue
                else:
                    new_seq.append(list(code_set))
        if len(new_seq) > 1:
            new_seqs.append(new_seq)

    # Building strSeqs, and string labels
    print('Building strSeqs, and string labels')
    new_seqs_str = []
    adm_dx_ccs = []
    adm_dx_ccs_cat1 = []
    for seq in new_seqs:
        seq_ls = []
        dx_ccs_ls = []
        dx_ccs_cat1_ls = []
        for adm in seq:
            new_adm = []
            dx_ccs = []
            dx_ccs_cat1 = []
            for dx in adm:
                dxStr = 'D_' + dx
                dxStr_ccs_single = 'D_' + label4data.code2single_dx[dx]
                dxStr_ccs_cat1 = 'D_' + label4data.code2first_level_dx[dx]
                new_adm.append(dxStr)
                dx_ccs.append(dxStr_ccs_single)
                dx_ccs_cat1.append(dxStr_ccs_cat1)
            seq_ls.append(new_adm)
            dx_ccs_ls.append(dx_ccs)
            dx_ccs_cat1_ls.append(dx_ccs_cat1)
        new_seqs_str.append(seq_ls)
        adm_dx_ccs.append(dx_ccs_ls)
        adm_dx_ccs_cat1.append(dx_ccs_cat1_ls)

    print('Converting strSeqs to intSeqs, and making types for ccs single-level code')
    dict_ccs = {}
    new_seqs_ccs = []
    for patient in adm_dx_ccs:
        new_patient = []
        for visit in patient:
            new_visit = []
            for code in set(visit):
                if code in dict_ccs:
                    new_visit.append(dict_ccs[code])
                else:
                    dict_ccs[code] = len(dict_ccs)
                    new_visit.append(dict_ccs[code])
            new_patient.append(new_visit)
        new_seqs_ccs.append(new_patient)

    print('Converting strSeqs to intSeqs, and making types for ccs multi-level first level code')
    dict_ccs_cat1 = {}
    new_seqs_ccs_cat1 = []
    for patient in adm_dx_ccs_cat1:
        new_patient = []
        for visit in patient:
            new_visit = []
            for code in set(visit):
                if code in dict_ccs_cat1:
                    new_visit.append(dict_ccs_cat1[code])
                else:
                    dict_ccs_cat1[code] = len(dict_ccs_cat1)
                    new_visit.append(dict_ccs_cat1[code])
            new_patient.append(new_visit)
        new_seqs_ccs_cat1.append(new_patient)

    print('Converting seqs to model inputs')
    inputs_all = []
    labels_ccs = []
    labels_next_visit = []
    labels_visit_cat1 = []
    vocab_set = {}
    for i, seq in enumerate(new_seqs_str):

        last_seqs = seq
        last_seq_ccs = new_seqs_ccs[i]
        last_seq_ccs_cat1 = new_seqs_ccs_cat1[i]

        valid_seq = last_seqs[:-1]

        labels_visit_cat1.append(last_seq_ccs_cat1[:-1])
        inputs_all.append(valid_seq)
        labels_ccs.append(last_seq_ccs[-1])
        labels_next_visit.append(last_seq_ccs[1:])

        for visit in valid_seq:
            for code in visit:
                if code in vocab_set:
                    vocab_set[code] += 1
                else:
                    vocab_set[code] = 1

    sorted_vocab = {k: v for k, v in sorted(vocab_set.items(), key=lambda item: item[1], reverse=True)}
    pickle.dump(inputs_all, open(out_file + 'inputs_all.seqs', 'wb'), -1)
    pickle.dump(labels_ccs, open(out_file + 'labels_ccs.label', 'wb'), -1)
    pickle.dump(labels_next_visit, open(out_file + 'labels_next_visit.label', 'wb'), -1)
    pickle.dump(labels_visit_cat1, open(out_file + 'labels_visit_cat1.label', 'wb'), -1)
    pickle.dump(dict_ccs, open(out_file + 'ccs_single_level.dict', 'wb'), -1)
    pickle.dump(dict_ccs_cat1, open(out_file + 'ccs_cat1.dict', 'wb'), -1)
    outfd = open(out_file + '.vocab.txt', 'w')
    for k, v in sorted_vocab.items():
        outfd.write(k + '\n')
    outfd.close()

    pickle.dump(new_seqs_str, open(out_file + 'origin.seqs', 'wb'), -1)
    pickle.dump(new_seqs_ccs, open(out_file + 'origin_ccs.seqs', 'wb'), -1)
    pickle.dump(new_seqs_ccs_cat1, open(out_file + 'origin_ccs_cat1.seqs', 'wb'), -1)

    max_seq_len = 0
    max_adm_len = 0
    for seq in new_seqs:
        if max_seq_len < len(seq):
            max_seq_len = len(seq)
        for adm in seq:
            if max_adm_len < len(adm):
                max_adm_len = len(adm)
    print(max_adm_len, max_seq_len, len(dict_ccs), len(inputs_all), len(sorted_vocab), len(dict_ccs_cat1))


if __name__ == '__main__':
    # dir_path = '../../../'
    # patient_file = dir_path + 'data/patient.csv'
    # dx_file = dir_path+ 'data/diagnosis.csv'
    # single_file = dir_path + 'ccs/ccs_single_dx_tool_2015.csv'
    # multi_file = dir_path + 'ccs/ccs_multi_dx_tool_2015.csv'
    # # out_file = dir_path + 'outputs/eICU/seq_prediction/eicu'
    # out_file = dir_path + 'outputs/eICU/statistics/eicu'

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="The output directory where the processed files will be written.")
    parser.add_argument("--patient",
                        type=str,
                        required=True,
                        help="The path of patient file.")
    parser.add_argument("--diagnosis",
                        type=str,
                        required=True,
                        help="The path of diagnosis file.")
    parser.add_argument("--single_level",
                        type=str,
                        required=True,
                        help="The path of CCS Single Level of diagnoses.")
    parser.add_argument("--multi_level",
                        type=str,
                        required=True,
                        help="The path of CCS multi-level of diagnoses.")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    processing_for_dx(args.patient, args.diagnosis, args.single_level,
                      args.multi_level, args.output)

