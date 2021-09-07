# SETOR

Source code and dataset for "Sequential Diagnosis Prediction with Transformer and Ontological Representation"

## Reqirements:

* Pytorch>=1.4.0
* Python3

## Data Preparation
### [MIMIC-III](https://physionet.org/content/mimiciii/1.4/)
```bash
data_mimic3_processing.py 
--output output_path/mimic/ 
--admission mimic_path/ADMISSIONS.csv  
--diagnosis mimic_path/DIAGNOSES_ICD.csv 
--single_level utils/ccs/ccs_single_dx_tool_2015.csv 
--multi_level utils/ccs/ccs_multi_dx_tool_2015.csv
```

### [eICU](https://physionet.org/content/eicu-crd/2.0/)
```bash
data_eicu_processing.py 
--output output_path/eicu/ 
--patient eicu_path/patient.csv  
--diagnosis eciu/diagnosis.csv
--single_level utils/ccs/ccs_single_dx_tool_2015.csv 
--multi_level utils/ccs/ccs_multi_dx_tool_2015.csv
```

### [MIMIC-IV](https://physionet.org/content/mimiciv/0.4/)
```bash
data_mimic4_processing.py 
--output ../outputs/order/data/mimic4/ 
--admission ../data/mimic_iv/admissions.csv.gz  
--diagnosis ../data/mimic_iv/diagnoses_icd.csv.gz 
--single_level utils/ccs/ccs_single_dx_tool_2015.csv 
--multi_level utils/ccs//ccs_multi_dx_tool_2015.csv 
--icd_convert utils/ccs//icd10cmtoicd9gem.csv
```

### Knowledge Graph Building

```bash
data_graph_building.py 
--output output_path/mimic/  
--seqs output_path/mimic/inputs_all.seqs 
--vocab output_path/mimic/vocab.txt 
--multi_level ../utils/ccs/ccs_multi_dx_tool_2015.csv
```

##  Model Training:Validating:Testing
```bash
train.py 
--output_dir ../outputs/order/model 
--data_dir ../outputs/order/data/ 
--num_train_epochs 10 
--train_batch_size 32 
--gpu 2 
--learning_rate 0.1
 --data_source eicu 
 --task dx_next 
 --train_ratio 0.8
```

