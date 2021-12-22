# NNIをインポートする
import nni
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder


def load_data(train_file_path):
    """
    データの前処理を行う関数
    Parameters
    ----------
    train_file_path : str
        学習用データのファイルパス
    Returns
    -------
    X_train : pd.DataFrame
        学習用のデータ
    y_train : Series
        学習用の正解ラベル
    """
    train_df = pd.read_csv(train_file_path)
    y_train = train_df.pop('Survived')
    X_train = train_df.drop(['PassengerId', 'Name'], axis=1)
    list_cols = ['Sex', 'Ticket', 'Cabin', 'Embarked']
    for col in list_cols:
        le = LabelEncoder()
        le.fit(X_train[col])
        X_train[col] = le.transform(X_train[col])
    return X_train, y_train


def get_default_parameters():
    """
    デフォルトのパラメーターを取得する関数
    Returns
    -------
    params : dict
        デフォルトのパラメーター
    """
    params = {
        'learning_rate': 0.02,
        'n_estimators': 2000,
        'max_depth': 4,
        'min_child_weight': 2,
        'gamma': 0.9,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'nthread': -1,
        'scale_pos_weight': 1
    }
    return params


def get_model(PARAMS):
    """
    モデルを入手する関数
    Parameters
    ----------
    PARAMS : dict
        パラメーター
    Returns
    -------
    model : xgboost.sklearn.XGBClassifier
        学習に使用するモデル
    """
    model = xgb.XGBClassifier()
    model.learning_rate = PARAMS.get("learning_rate")
    model.max_depth = PARAMS.get("max_depth")
    model.subsample = PARAMS.get("subsample")
    model.colsample_btree = PARAMS.get("colsample_btree")
    return model


def run(X_train, y_train, model):
    """
    モデルを実行する関数
    Parameters
    ----------
    X_train : pd.DataFrame
        学習用のデータ
    y_train : pd.DataFrame
        学習用の正解ラベル
    model : xgboost.sklearn.XGBClassifier
        学習に使用するモデル
    """
    scores = cross_val_score(model, X_train, y_train,
                             scoring='accuracy', cv=KFold(n_splits=5))
    score = scores.mean()
    # Configurationの結果を報告する
    nni.report_final_result(score)


if __name__ == '__main__':
    X_train_sub, y_train_sub = load_data('nni/train.csv')
    # TunerからConfigurationを取得する
    RECEIVED_PARAMS = nni.get_next_parameter()
    PARAMS = get_default_parameters()
    PARAMS.update(RECEIVED_PARAMS)
    model = get_model(PARAMS)
    run(X_train_sub, y_train_sub, model)
