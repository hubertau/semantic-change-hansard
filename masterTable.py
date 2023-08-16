import pandas as pd

from VanillaModel.vanillamodel import vanilla
from WeightedSimilarityExperiments.weighting_experiment import weighting
from BaselineModel.baselinemptimemodel import mpTime
from Retrofitting.prePostRetrofitting.eval_retrofitting import retrofitting


masterTable = pd.concat([vanilla(),mpTime(),weighting(), retrofitting('party'), retrofitting('partyTime')])
masterTable.columns = ['MODEL', 'DESCRIPTION', 'BASIS', 'ACCURACY', 'PRECISION', 'RECALL', 'F1 SCORE']
masterTable = masterTable.sort_values(by='PRECISION', ascending=False)
print(masterTable)
