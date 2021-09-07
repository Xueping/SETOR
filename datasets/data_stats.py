import pickle

dir_path = '../../../'

mimic_file = dir_path + 'outputs/kemce/data/statistics/'
mimic_ccs = mimic_file + 'mimic.origin_ccs.seqs'
mimic_ccs_cat1 = mimic_file + 'mimic.origin_ccs_cat1.seqs'
mimic_ccs_dict = mimic_file + 'mimic.ccs_single_level.dict'
mimic_ccs_cat1_dict = mimic_file + 'mimic.ccs_cat1.dict'

mimic_ccs_seqs = pickle.load(open(mimic_ccs, 'rb'))
mimic_ccs_cat1_seqs = pickle.load(open(mimic_ccs_cat1, 'rb'))
mimic_ccs_dict = pickle.load(open(mimic_ccs_dict, 'rb'))
mimic_ccs_cat1_dict = pickle.load(open(mimic_ccs_cat1_dict, 'rb'))

print(len(mimic_ccs_seqs), len(mimic_ccs_cat1_seqs), len(mimic_ccs_dict), len(mimic_ccs_cat1_dict))
num_visits = 0
total_codes = 0
max_len_codes = 0
for seq in mimic_ccs_seqs:
    num_visits += len(seq)
    for visit in seq:
        len_visit = len(visit)
        total_codes += len_visit
        if len_visit > max_len_codes:
            max_len_codes = len_visit
print('MIMIC III category codes:')
print(num_visits, total_codes, max_len_codes, total_codes/num_visits)

num_visits = 0
total_codes = 0
max_len_codes = 0
for seq in mimic_ccs_cat1_seqs:
    num_visits += len(seq)
    for visit in seq:
        len_visit = len(visit)
        total_codes += len_visit
        if len_visit > max_len_codes:
            max_len_codes = len_visit
print('MIMIC III First-level category codes:')
print(num_visits, total_codes, max_len_codes, total_codes/num_visits)

# for eICU datasets
eicu_file = dir_path + 'outputs/data/eICU/statistics/'
eicu = eicu_file + 'eicu.origin.seqs'
eicu_ccs = eicu_file + 'eicu.origin_ccs.seqs'
eicu_ccs_cat1 = eicu_file + 'eicu.origin_ccs_cat1.seqs'
eicu_ccs_dict = eicu_file + 'eicu.ccs_single_level.dict'
eicu_ccs_cat1_dict = eicu_file + 'eicu.ccs_cat1.dict'

eicu_seqs = pickle.load(open(eicu, 'rb'))
eicu_ccs_seqs = pickle.load(open(eicu_ccs, 'rb'))
eicu_ccs_cat1_seqs = pickle.load(open(eicu_ccs_cat1, 'rb'))
eicu_ccs_dict = pickle.load(open(eicu_ccs_dict, 'rb'))
eicu_ccs_cat1_dict = pickle.load(open(eicu_ccs_cat1_dict, 'rb'))

print('eicu statistics:')
print(len(eicu_ccs_seqs), len(eicu_ccs_cat1_seqs), len(eicu_ccs_dict), len(eicu_ccs_cat1_dict))
num_visits = 0
total_codes = 0
max_len_codes = 0
for seq in eicu_ccs_seqs:
    num_visits += len(seq)
    for visit in seq:
        len_visit = len(visit)
        total_codes += len_visit
        if len_visit > max_len_codes:
            max_len_codes = len_visit
print('eicu category codes:')
print(num_visits, total_codes, max_len_codes, total_codes/num_visits)

num_visits = 0
total_codes = 0
max_len_codes = 0
for seq in eicu_ccs_cat1_seqs:
    num_visits += len(seq)
    for visit in seq:
        len_visit = len(visit)
        total_codes += len_visit
        if len_visit > max_len_codes:
            max_len_codes = len_visit
print('eicu First-level category codes:')
print(num_visits, total_codes, max_len_codes, total_codes/num_visits)

num_patients = len(eicu_seqs)
all_codes = []
num_visits = 0
total_codes = 0
max_len_codes = 0
for seq in eicu_seqs:
    num_visits += len(seq)
    for visit in seq:
        all_codes.extend(visit)
        len_visit = len(visit)
        total_codes += len_visit
        if len_visit > max_len_codes:
            max_len_codes = len_visit
print('eicu ICD9 code codes:')
print(num_visits, num_visits/num_patients, max_len_codes, total_codes/num_visits,len(set(all_codes)))


# for eICU datasets
eicu_file = dir_path + 'outputs/data/MIMIC_IV/statistics/'
eicu = eicu_file + 'origin.seqs'
eicu_ccs = eicu_file + 'origin_ccs.seqs'
eicu_ccs_cat1 = eicu_file + 'origin_ccs_cat1.seqs'
eicu_ccs_dict = eicu_file + 'ccs_single_level.dict'
eicu_ccs_cat1_dict = eicu_file + 'ccs_cat1.dict'

eicu_seqs = pickle.load(open(eicu, 'rb'))
eicu_ccs_seqs = pickle.load(open(eicu_ccs, 'rb'))
eicu_ccs_cat1_seqs = pickle.load(open(eicu_ccs_cat1, 'rb'))
eicu_ccs_dict = pickle.load(open(eicu_ccs_dict, 'rb'))
eicu_ccs_cat1_dict = pickle.load(open(eicu_ccs_cat1_dict, 'rb'))

print('MIMIC-IV statistics:')
print(len(eicu_ccs_seqs), len(eicu_ccs_cat1_seqs), len(eicu_ccs_dict), len(eicu_ccs_cat1_dict))
num_visits = 0
total_codes = 0
max_len_codes = 0
for seq in eicu_ccs_seqs:
    num_visits += len(seq)
    for visit in seq:
        len_visit = len(visit)
        total_codes += len_visit
        if len_visit > max_len_codes:
            max_len_codes = len_visit
print('MIMIC-IV category codes:')
print(num_visits, total_codes, max_len_codes, total_codes/num_visits)

num_visits = 0
total_codes = 0
max_len_codes = 0
for seq in eicu_ccs_cat1_seqs:
    num_visits += len(seq)
    for visit in seq:
        len_visit = len(visit)
        total_codes += len_visit
        if len_visit > max_len_codes:
            max_len_codes = len_visit
print('MIMIC-IV First-level category codes:')
print(num_visits, total_codes, max_len_codes, total_codes/num_visits)

num_patients = len(eicu_seqs)
all_codes = []
num_visits = 0
total_codes = 0
max_len_codes = 0
for seq in eicu_seqs:
    num_visits += len(seq)
    for visit in seq:
        all_codes.extend(visit)
        len_visit = len(visit)
        total_codes += len_visit
        if len_visit > max_len_codes:
            max_len_codes = len_visit
print('MIMIC-IV ICD9 code codes:')
print(num_visits, num_visits/num_patients, max_len_codes, total_codes/num_visits,len(set(all_codes)))




