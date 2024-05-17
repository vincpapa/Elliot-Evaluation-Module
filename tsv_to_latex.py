import pandas as pd
import glob

dataset = 'amazon_music'
recs = glob.glob(f'results/{dataset}/performance/rec_cutoff_20*26*')
for rec in recs:
    df = pd.read_csv(rec, sep='\t')
    df = df[['model', 'nDCG', 'Recall', 'Gini', 'ItemCoverage', 'APLT', 'PopRSP', 'UserMADranking_ActivityLevel', 'PDU_m0:nDCG-m1:APLT']]
    df = df.round(4)
    print(df.to_latex())

