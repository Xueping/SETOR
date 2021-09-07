import pandas as pd


class LabelsForData(object):

    def __init__(self, ccs_multi_dx_file, ccs_single_dx_file):
        self.ccs_multi_dx_df = pd.read_csv(ccs_multi_dx_file, header=0, dtype=object)
        self.ccs_single_dx_df = pd.read_csv(ccs_single_dx_file, header=0, dtype=object)
        self.code2single_dx = {}  # label sequential diagnosis prediction data
        self.code2first_level_dx = {}  # label clustering data
        self.build_maps()

    def build_maps(self):
        for i, row in self.ccs_multi_dx_df.iterrows():
            # print(row)
            code = row[0][1:-1].strip()
            level_1_cat = row[1][1:-1].strip()
            self.code2first_level_dx[code] = level_1_cat

        for i, row in self.ccs_single_dx_df.iterrows():
            code = row[0][1:-1].strip()
            single_cat = row[1][1:-1].strip()
            self.code2single_dx[code] = single_cat

