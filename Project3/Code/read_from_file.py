import os
import random

import numpy as np
import pandas as pd

# Trying to set the seed
np.random.seed(0)
random.seed(0)

# Reading file into data frame
cwd = os.getcwd()
population_path = cwd + '/../Data/ASV_table.tsv'
metadata_path = cwd + '/../Data/Metadata_table.tsv'
nanDict = {}
"""
df.columns (identifier)
df.values (population size)

population_size.shape -> (72, 14991)
"""
population_size = pd.read_csv(population_path, delimiter='\s+', encoding='utf-8')

"""
df.columns (properties)
df.values (values)

metadata.shape -> (71, 41)
"""
metadata = pd.read_csv(metadata_path, delimiter='\s+', encoding='utf-8')



