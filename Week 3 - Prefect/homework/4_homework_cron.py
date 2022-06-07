import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from prefect import task, flow, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from datetime import datetime
import pickle
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner
import os

@task
def get_paths(date: str):
    if date == None:
        datetime_obj = datetime.today()
    else:
        datetime_obj = datetime.strptime(date, '%Y-%m-%d')
    logger = get_run_logger()
    train_path = f'./data/fhv_tripdata_{datetime_obj.year}-{(datetime_obj.month - 2):02d}.parquet'
    val_path = f'./data/fhv_tripdata_{datetime_obj.year}-{(datetime_obj.month - 1):02d}.parquet'
    logger.info(f'Train file: {train_path}')
    logger.info(f'Validation file: {val_path}')
    return train_path, val_path

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@flow(task_runner=SequentialTaskRunner())
#def main(date="2021-03-15"):
def main(date="2021-08-15"):
    logger = get_run_logger()
    train_path, val_path = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, train=False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    model_path = f'bins/models/model-{date}.bin'
    with open(model_path, 'wb') as f_out:
        pickle.dump(lr, f_out)
    logger.info(f'Model saved in: {model_path}')

    preprocessor_path = f'bins/preprocessors/dv-{date}.b'
    with open(preprocessor_path, 'wb') as f_out:
        pickle.dump(dv, f_out)
    logger.info(f'Preprocessor saved in: {preprocessor_path}')

# DeploymentSpec
DeploymentSpec(
        name="cron-schedule-deployment",
        #flow_location="/path/to/flow.py",
        flow = main,
        flow_runner=SubprocessFlowRunner(),
        schedule=CronSchedule(
            cron="0 9 15 * *",
            timezone="America/Sao_Paulo"),
    )

if __name__ == "__main__":
    main()
    model_list = os.listdir('bins/models')
    preprocessor_list = os.listdir('bins/preprocessors')
    print('Latest saved model: ', sorted(model_list, reverse=False)[0])
    print('Latest saved preprocessor: ', sorted(preprocessor_list, reverse=False)[0])
