import pandas
import os
import numpy as np
import json

def load_metadata(metadata_dir,metadata_type="json"):
    if metadata_type=="csv":
      df = pandas.read_csv(os.path.join(metadata_dir,"metadata.csv"))
      #create a dictionary with filename as key and label as value
      metadata = {df.iloc[i,0]:df.iloc[i,1] for i in range(len(df))}
    elif metadata_type=="json":
      #load json file
      metadata = {}
      df = json.load(open(os.path.join(metadata_dir,"metadata.json")))
      for k,v in df.items():
        metadata[k] = v["label"]
    return metadata
  

def convert_json_csv(metadata_path):
    metadata = json.load(open(metadata_path))
    metadata = pd.DataFrame(metadata).transpose()
    metadata.reset_index(level=0, inplace=True)
    metadata.rename(columns={'index':'filename'}, inplace=True)
    return metadata

      