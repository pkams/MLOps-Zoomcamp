from datetime import datetime

def get_paths(date: str):
    if date == None:
        datetime_obj = datetime.today()
    else:
        datetime_obj = datetime.strptime(date, '%Y-%m-%d')
    train_path = f'./data/fhv_tripdata_{datetime_obj.year}-{(datetime_obj.month - 2):02d}.parquet'
    val_path = f'./data/fhv_tripdata_{datetime_obj.year}-{(datetime_obj.month - 1):02d}.parquet'
    return train_path, val_path

print(get_paths(None))
print(get_paths('2021-03-15'))