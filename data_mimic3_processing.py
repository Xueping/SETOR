from datetime import datetime
import pickle
import argparse
import os
from datasets.data_labelling import LabelsForData


def mimic_processing(adm_file, dx_file, single_dx_file, multi_dx_file, out_file):

    label4data = LabelsForData(multi_dx_file, single_dx_file)
    print('Building pid-admission mapping, admission-date mapping')
    pidAdmMap = {}
    admDateMap = {}
    infd = open(adm_file, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[1])
        admId = int(tokens[2])
        admTime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
        admDateMap[admId] = admTime
        if pid in pidAdmMap:
            pidAdmMap[pid].append(admId)
        else:
            pidAdmMap[pid] = [admId]
    infd.close()

    print('Building admission-dxList mapping')
    admDxMap = {}
    admDxMap_ccs = {}
    admDxMap_ccs_cat1 = {}
    infd = open(dx_file, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        admId = int(tokens[2])
        dx = tokens[4][1:-1]
        if len(dx) == 0:
            continue

        dxStr = 'D_' + dx
        dxStr_ccs_single = 'D_' + label4data.code2single_dx[dx]
        dxStr_ccs_cat1 = 'D_' + label4data.code2first_level_dx[dx]

        if admId in admDxMap:
            admDxMap[admId].append(dxStr)
        else:
            admDxMap[admId] = [dxStr]

        if admId in admDxMap_ccs:
            admDxMap_ccs[admId].append(dxStr_ccs_single)
        else:
            admDxMap_ccs[admId] = [dxStr_ccs_single]

        if admId in admDxMap_ccs_cat1:
            admDxMap_ccs_cat1[admId].append(dxStr_ccs_cat1)
        else:
            admDxMap_ccs_cat1[admId] = [dxStr_ccs_cat1]
    infd.close()

    print('Building pid-sortedVisits mapping')
    pidSeqMap = {}
    pidSeqMap_ccs = {}
    pidSeqMap_ccs_cat1 = {}
    for pid, admIdList in pidAdmMap.items():
        new_admIdList = []
        for admId in admIdList:
            if admId in admDxMap:
                new_admIdList.append(admId)
        if len(new_admIdList) < 2: continue
        # print(admIdList)
        sortedList = sorted([(admDateMap[admId], admDxMap[admId]) for admId in new_admIdList])
        pidSeqMap[pid] = sortedList

        sortedList_ccs = sorted([(admDateMap[admId], admDxMap_ccs[admId]) for admId in new_admIdList])
        pidSeqMap_ccs[pid] = sortedList_ccs

        sortedList_ccs_cat1 = sorted([(admDateMap[admId], admDxMap_ccs_cat1[admId]) for admId in new_admIdList])
        pidSeqMap_ccs_cat1[pid] = sortedList_ccs_cat1

    print('Building strSeqs, span label')
    seqs = []
    seqs_span = []
    seqs_intervals = []
    for pid, visits in pidSeqMap.items():
        seq = []
        spans = []
        intervals = []
        first_time = visits[0][0]
        for i, visit in enumerate(visits):
            current_time = visit[0]
            interval = (current_time - first_time).days
            first_time = current_time
            seq.append(visit[1])
            span_flag = 0 if interval <= 30 else 1
            spans.append(span_flag)
            intervals.append(interval)
        seqs.append(seq)
        seqs_span.append(spans)
        seqs_intervals.append(intervals)

    print('Building strSeqs for CCS single-level code')
    seqs_ccs = []
    for pid, visits in pidSeqMap_ccs.items():
        seq = []
        for visit in visits:
            seq.append(visit[1])
        seqs_ccs.append(seq)

    print('Converting strSeqs to intSeqs, and making types for ccs single-level code')
    dict_ccs = {}
    newSeqs_ccs = []
    for patient in seqs_ccs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in set(visit):
                if code in dict_ccs:
                    newVisit.append(dict_ccs[code])
                else:
                    dict_ccs[code] = len(dict_ccs)
                    newVisit.append(dict_ccs[code])
            newPatient.append(newVisit)
        newSeqs_ccs.append(newPatient)

    print('Building strSeqs for CCS multi-level first code')
    seqs_ccs_cat1 = []
    for pid, visits in pidSeqMap_ccs_cat1.items():
        seq = []
        for visit in visits:
            seq.append(visit[1])
        seqs_ccs_cat1.append(seq)

    print('Converting strSeqs to intSeqs, and making types for ccs multi-level first level code')
    dict_ccs_cat1 = {}
    newSeqs_ccs_cat1 = []
    for patient in seqs_ccs_cat1:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in set(visit):
                if code in dict_ccs_cat1:
                    newVisit.append(dict_ccs_cat1[code])
                else:
                    dict_ccs_cat1[code] = len(dict_ccs_cat1)
                    newVisit.append(dict_ccs_cat1[code])
            newPatient.append(newVisit)
        newSeqs_ccs_cat1.append(newPatient)

    print('Converting seqs to model inputs')
    inputs_all = []
    labels_ccs = []
    intervals_all = []
    labels_current_visit = []
    labels_next_visit = []
    labels_visit_cat1 = []
    vocab_set = {}
    max_visit_len = 0
    max_seqs_len = 0
    truncated_len = 21
    for i, seq in enumerate(seqs):
        length = len(seq)

        if length >= truncated_len:
            last_seqs = seq[length-truncated_len:]
            last_spans = seqs_span[i][length-truncated_len:]
            last_seq_ccs = newSeqs_ccs[i][length-truncated_len:]
            last_seq_ccs_cat1 = newSeqs_ccs_cat1[i][length-truncated_len:]
            last_seq_intervals = seqs_intervals[i][length - truncated_len:]
        else:
            last_seqs = seq
            last_spans = seqs_span[i]
            last_seq_ccs = newSeqs_ccs[i]
            last_seq_ccs_cat1 = newSeqs_ccs_cat1[i]
            last_seq_intervals = seqs_intervals[i]

        valid_seq = last_seqs[:-1]
        label_span = last_spans[-1]
        label_ccs = last_seq_ccs[-1]
        label_current_visit = last_seq_ccs[:-1]
        label_next_visit = last_seq_ccs[1:]
        valid_intervals = last_seq_intervals[:-1]

        labels_current_visit.append(label_current_visit)
        labels_next_visit.append(label_next_visit)
        labels_visit_cat1.append(last_seq_ccs_cat1[:-1])
        inputs_all.append(valid_seq)
        labels_ccs.append((label_ccs))
        intervals_all.append(valid_intervals)

        if len(valid_seq) > max_seqs_len:
            max_seqs_len = len(valid_seq)

        for visit in valid_seq:
            if len(visit) > max_visit_len:
                max_visit_len = len(visit)
            for code in visit:
                if code in vocab_set:
                    vocab_set[code] += 1
                else:
                    vocab_set[code] = 1

    pickle.dump(inputs_all, open(os.path.join(out_file,'inputs_all.seqs'), 'wb'), -1)
    pickle.dump(intervals_all, open(os.path.join(out_file, 'intervals_all.seqs'), 'wb'), -1)
    pickle.dump(labels_ccs, open(os.path.join(out_file, 'labels_ccs.label'), 'wb'), -1)
    pickle.dump(labels_current_visit, open(os.path.join(out_file, 'labels_current_visit.label'), 'wb'), -1)
    pickle.dump(labels_next_visit, open(os.path.join(out_file, 'labels_next_visit.label'), 'wb'), -1)
    pickle.dump(labels_visit_cat1, open(os.path.join(out_file, 'labels_visit_cat1.label'), 'wb'), -1)
    pickle.dump(dict_ccs, open(os.path.join(out_file, 'ccs_single_level.dict'), 'wb'), -1)
    pickle.dump(dict_ccs_cat1, open(os.path.join(out_file, 'ccs_cat1.dict'), 'wb'), -1)

    pickle.dump(seqs, open(os.path.join(out_file, 'origin.seqs'), 'wb'), -1)
    pickle.dump(newSeqs_ccs, open(os.path.join(out_file, 'origin_ccs.seqs'), 'wb'), -1)
    pickle.dump(newSeqs_ccs_cat1, open(os.path.join(out_file, 'origin_ccs_cat1.seqs'), 'wb'), -1)

    sorted_vocab = {k: v for k, v in sorted(vocab_set.items(), key=lambda item: item[1], reverse=True)}
    outfd = open(os.path.join(out_file, 'vocab.txt'), 'w')
    for k, v in sorted_vocab.items():
        outfd.write(k + '\n')
    outfd.close()
    print(max_visit_len, max_seqs_len, len(dict_ccs), len(inputs_all), len(sorted_vocab), len(dict_ccs_cat1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="The output directory where the processed files will be written.")
    parser.add_argument("--admission",
                        type=str,
                        required=True,
                        help="The path of admission file.")
    parser.add_argument("--diagnosis",
                        type=str,
                        required=True,
                        help="The path of diagnosis file.")
    parser.add_argument("--single_level",
                        type=str,
                        required=True,
                        help="The path of CCS Single Level of diagnoses.")
    parser.add_argument("--multi_level",
                        # default=None,
                        type=str,
                        required=True,
                        help="The path of CCS multi-level of diagnoses.")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    mimic_processing(args.admission, args.diagnosis, args.single_level,
                     args.multi_level, args.output)

