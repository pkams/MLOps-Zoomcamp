import pickle
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year", type=str, default = '2021')
parser.add_argument("-m", "--month", type=str, default = '02')
parser.add_argument("-s", "--save", type=bool, default = False)
args = parser.parse_args()

year = args.year
month = args.month
save = args.save

output_file = f'output/fhv_tripdata_{year}-{month}-results.parquet'

os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)

categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year}-{month}.parquet')
df['ride_id'] = f'{year}/{month}_' + df.index.astype('str')
dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)
print(f'Mean of predicted duration: {y_pred.mean()}')
print('Preparando para salvar...')
if save:
    df['predictions'] = y_pred
    df_result = df[['ride_id', 'predictions']]
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    df_result.to_csv(output_file.replace('parquet', 'csv'),sep=';', index=None)
    print('Salvou.')