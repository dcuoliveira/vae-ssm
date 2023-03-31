import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler

def hyper_params_search(df,
                        wrapper,
                        target_name,
                        n_iter,
                        n_splits,
                        n_jobs,
                        verbose,
                        seed):
    """
    Use the dataframe 'df' to search for the best
    params for the model 'wrapper'.
    The CV split is performed using the TimeSeriesSplit
    class.
    We can define the size of the test set using the formula
    ``n_samples//(n_splits + 1)``,
    where ``n_samples`` is the number of samples. Hence,
    we can define
    n_splits = (n - test_size) // test_size

    :param df: train data
    :type df: pd.DataFrame
    :param wrapper: predictive model
    :type wrapper: sklearn model wrapper
    :param target_name: name of the target column in 'df'
    :type target_name: str
    :param n_iter: number of hyperparameter searchs
    :type n_iter: int
    :param n_splits: number of splits for the cross-validation
    :type n_splits: int
    :param n_jobs: number of concurrent workers
    :type n_jobs: int
    :param verbose: param to print iteration status
    :type verbose: bool, int
    :return: R2 value
    :rtype: float
    """

    X = df.drop(labels=target_name, axis=1).values
    y = df[target_name].values
    scoring = make_scorer(mean_squared_error)

    time_split = TimeSeriesSplit(n_splits=n_splits)

    if wrapper.search_type == 'random':
        model_search = RandomizedSearchCV(estimator=wrapper.ModelClass,
                                          param_distributions=wrapper.param_grid,
                                          n_iter=n_iter,
                                          cv=time_split,
                                          verbose=verbose,
                                          n_jobs=n_jobs,
                                          scoring=scoring,
                                          random_state=seed)
    elif wrapper.search_type == 'grid':
        model_search = GridSearchCV(estimator=wrapper.ModelClass,
                                    param_grid=wrapper.param_grid,
                                    cv=time_split,
                                    verbose=verbose,
                                    n_jobs=n_jobs,
                                    scoring=scoring)
    else:
        raise Exception('search type method not registered')

    model_search = model_search.fit(X, y)

    return model_search


def train_model(df,
                init_steps,
                predict_steps,
                Wrapper,
                target_name,
                n_iter,
                n_splits,
                n_jobs,
                verbose,
                seed):
    """
     Star the training procedure considering "init_steps" as the estimate
     starating point.
     We recursively increase the training sample, periodically refitting
     the entire model once per "predict_steps", and making
     out-of-sample predictions for the subsequent "predict_steps".
     On each fit, to perform hyperparameter search,
     we perform cross-validation on a rolling basis.

     :param df: train and test data combined
     :type df: pd.DataFrame
     :param init_steps: number of observations to use as starting point for the estimation
     :type init_steps: int
     :param predict_steps: number of steps ahead to predict
     :type predict_steps: int
     :param Wrapper: predictive model class
     :type Wrapper: sklearn model wrapper class
     :param n_iter: number of hyperparameter searchs
     :type n_iter: int
     :param n_splits: number of splits for the cross-validation
     :type n_splits: int
     :param n_jobs: number of concurrent workers
     :type n_jobs: int
     :param verbose: param to print iteration status
     :type verbose: bool, int
     :param seed: seed for the random hyperparameter search
     :type seed: int
     :param target_name: name of the target column in 'df'
     :type target_name: str
     :return: dataframe with the date, true return
              and predicted return.
     :rtype: pd.DataFrame
     """

    all_preds = []
    for t in tqdm(range(init_steps, df.shape[0] - predict_steps, predict_steps), desc="Running TSCV"):
        
        train_df = df[:t]
        test_df = df[t:(t + predict_steps)]

        # NOTE - Medeiros et al. (2019) do not apply any scaler technique on the inflation data 
        # scaler = StandardScaler()
        # train_df = pd.DataFrame(scaler.fit_transform(train_df),
        #                         columns=df[:t].columns,
        #                         index=df[:t].index)
        # test_df = pd.DataFrame(scaler.transform(test_df),
        #                        columns=df[t:(t + predict_steps)].columns,
        #                        index=df[t:(t + predict_steps)].index)

        model_wrapper = Wrapper()
        model_search = hyper_params_search(df=train_df,
                                           wrapper=model_wrapper,
                                           target_name=target_name,
                                           n_jobs=n_jobs,
                                           n_splits=n_splits,
                                           n_iter=n_iter,
                                           seed=seed,
                                           verbose=verbose)
        X_test = test_df.drop(labels=target_name, axis=1).values
        test_pred = model_search.best_estimator_.predict(X_test)
        
        dict_ = {
            "date": test_df.index,
            "true": test_df[target_name],
            "prediction": test_pred.item()
            }
        
        result = pd.DataFrame(dict_)
        all_preds.append(result)

    results = pd.concat(all_preds).reset_index(drop=True)

    return results