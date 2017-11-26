"""Train the models
"""
from __future__ import absolute_import
from __future__ import print_function
from keras.callbacks import EarlyStopping, History, TensorBoard, ModelCheckpoint
from keras.models import load_model
import hyperopt
from hyperopt.utils import coarse_utcnow
from hyperopt.mongoexp import MongoTrials
import kopt.eval_metrics as ce
from kopt.utils import write_json, merge_dicts, _to_string
from kopt.model_data import (subset, split_train_test_idx, split_KFold_idx)
from kopt.config import db_host, db_port, save_dir
from datetime import datetime, timedelta
from uuid import uuid4
from hyperopt import STATUS_OK
import numpy as np
import pandas as pd
from copy import deepcopy
import os
import glob
import pprint
import logging


logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def test_fn(fn, hyper_params, n_train=1000, save_model='best', tmp_dir="/tmp/kopt_test/", custom_objects=None):
    """Test the correctness of the compiled objective function (CompileFN). I will also test
    model saving/loading from disk.

    # Arguments
        fn: CompileFN instance
        hyper_params: pyll graph of hyper-parameters - as later provided to `hyperopt.fmin`
        n_train: int, number of training points
        tmp_dir: Temporary path where to write the trained model.
        save_model: If not None, the trained model is saved to a temporary directory
                    If save_model="best", save the best model using `keras.callbacks.ModelCheckpoint`, and
                    if save_model="last", save the model after training it.
        custom_objects: argument passed to load_model - Optional dictionary mapping names (strings) to
             custom classes or functions to be considered during deserialization.
    """
    def wrap_data_fn(data_fn, n_train=100):
        def new_data_fn(*args, **kwargs):
            data = data_fn(*args, **kwargs)
            train = data[0]
            train = subset(train, idx=np.arange(min(n_train, train[1].shape[0])))
            return train,
        return new_data_fn
    start_time = datetime.now()
    fn = deepcopy(fn)
    hyper_params = deepcopy(hyper_params)
    fn.save_dir = tmp_dir
    fn.save_model = save_model
    fn.data_fn = wrap_data_fn(fn.data_fn, n_train)

    # sample from hyper_params
    param = hyperopt.pyll.stochastic.sample(hyper_params)
    # overwrite the number of epochs
    if param.get("fit") is None:
        param["fit"] = {}
    param["fit"]["epochs"] = 1

    # correct execution
    res = fn(param)
    print("Returned value:")
    pprint.pprint(res)
    assert res["status"] == STATUS_OK

    if save_model:
        # correct model loading
        model_path = max(glob.iglob(fn.save_dir_exp + '/train_models/*.h5'),
                         key=os.path.getctime)
        assert datetime.fromtimestamp(os.path.getctime(model_path)) > start_time
        load_model(model_path, custom_objects=custom_objects)


class KMongoTrials(MongoTrials):
    """`hyperopt.MonoTrials` extended with the following methods:

    - get_trial(tid) - Retrieve trial by tid (Trial ID).
    - get_param(tid) - Retrieve used hyper-parameters for a trial.
    - best_trial_tid(rank=0) - Return the trial with lowest loss.
            - rank - rank=0 means the best model, rank=1 means second best, ...
    - optimal_epochs(tid) - Number of optimal epochs (after early-stopping)
    - delete_running(timeout_last_refresh=0, dry_run=False) - Delete jobs stalled in the running state for too long
            - timeout_last_refresh, int: number of seconds
            - dry_run, bool: If True, just simulate the removal but don't actually perform it.
    - valid_tid() - List all valid tid's
    - train_history(tid=None) - Get train history as pd.DataFrame with columns: `(epoch, loss, val_loss, ...)`
            - tid: Trial ID or list of trial ID's. If None, report for all trial ID's.
    - get_ok_results - Return a list of trial results with an "ok" status
    - load_model(tid) - Load a Keras model of a tid.
    - as_df - Returns a tidy `pandas.DataFrame` of the trials database.

    # Arguments
        db_name: str, MongoTrials database name
        exp_name: strm, MongoTrials experiment name
        ip: str, MongoDB IP address.
        port: int, MongoDB port.
        kill_timeout: int, Maximum runtime of a job (in seconds) before it gets killed. None for infinite.
        **kwargs: Additional keyword arguments passed to the `hyperopt.MongoTrials` constructor.

    """

    def __init__(self, db_name, exp_name,
                 ip=db_host(), port=db_port(), kill_timeout=None, **kwargs):
        self.kill_timeout = kill_timeout
        if self.kill_timeout is not None and self.kill_timeout < 60:
            logger.warning("kill_timeout < 60 -> Very short time for " +
                           "each job to complete before it gets killed!")

        super(KMongoTrials, self).__init__(
            'mongo://{ip}:{p}/{n}/jobs'.format(ip=ip, p=port, n=db_name), exp_key=exp_name, **kwargs)

    def get_trial(self, tid):
        """Retrieve trial by tid
        """
        lid = np.where(np.array(self.tids) == tid)[0][0]
        return self.trials[lid]

    def get_param(self, tid):
        # TODO - return a dictionary - add .to_dict()
        return self.get_trial(tid)["result"]["param"]

    def best_trial_tid(self, rank=0):
        """Get tid of the best trial

        rank=0 means the best model
        rank=1 means second best
        ...
        """
        candidates = [t for t in self.trials
                      if t['result']['status'] == STATUS_OK]
        if len(candidates) == 0:
            return None
        losses = [float(t['result']['loss']) for t in candidates]
        assert not np.any(np.isnan(losses))
        lid = np.where(np.argsort(losses).argsort() == rank)[0][0]
        return candidates[lid]["tid"]

    def optimal_epochs(self, tid):
        trial = self.get_trial(tid)
        patience = trial["result"]["param"]["fit"]["patience"]
        epochs = trial["result"]["param"]["fit"]["epochs"]

        def optimal_len(hist):
            c_epoch = max(hist["loss"]["epoch"]) + 1
            if c_epoch == epochs:
                return epochs
            else:
                return c_epoch - patience

        hist = trial["result"]["history"]
        if isinstance(hist, list):
            return int(np.floor(np.array([optimal_len(h) for h in hist]).mean()))
        else:
            return optimal_len(hist)

    # def refresh(self):
    #     """Extends the original object
    #     """
    #     self.refresh_tids(None)
    #     if self.kill_timeout is not None:
    #         # TODO - remove dry_run
    #         self.delete_running(self.kill_timeout, dry_run=True)

    def count_by_state_unsynced(self, arg):
        """Extends the original object in order to inject checking
        for stalled jobs and killing them if they are running for too long
        """
        if self.kill_timeout is not None:
            self.delete_running(self.kill_timeout)
        return super(KMongoTrials, self).count_by_state_unsynced(arg)

    def delete_running(self, timeout_last_refresh=0, dry_run=False):
        """Delete jobs stalled in the running state for too long

        timeout_last_refresh, int: number of seconds
        """
        running_all = self.handle.jobs_running()
        running_timeout = [job for job in running_all
                           if coarse_utcnow() > job["refresh_time"] +
                           timedelta(seconds=timeout_last_refresh)]
        if len(running_timeout) == 0:
            # Nothing to stop
            self.refresh_tids(None)
            return None

        if dry_run:
            logger.warning("Dry run. Not removing anything.")

        logger.info("Removing {0}/{1} running jobs. # all jobs: {2} ".
                    format(len(running_timeout), len(running_all), len(self)))

        now = coarse_utcnow()
        logger.info("Current utc time: {0}".format(now))
        logger.info("Time horizont: {0}".format(now - timedelta(seconds=timeout_last_refresh)))
        for job in running_timeout:
            logger.info("Removing job: ")
            pjob = job.to_dict()
            del pjob["misc"]  # ignore misc when printing
            logger.info(pprint.pformat(pjob))
            if not dry_run:
                self.handle.delete(job)
                logger.info("Job deleted")
        self.refresh_tids(None)

    # def delete_trial(self, tid):
    #     trial = self.get_trial(tid)
    #     return self.handle.delete(trial)

    def valid_tid(self):
        """List all valid tid's
        """
        return [t["tid"] for t in self.trials if t["result"]["status"] == "ok"]

    def train_history(self, tid=None):
        """Get train history as pd.DataFrame
        """

        def result2history(result):
            if isinstance(result["history"], list):
                return pd.concat([pd.DataFrame(hist["loss"]).assign(fold=i)
                                  for i, hist in enumerate(result["history"])])
            else:
                return pd.DataFrame(result["history"]["loss"])

        # use all
        if tid is None:
            tid = self.valid_tid()

        res = [result2history(t["result"]).assign(tid=t["tid"]) for t in self.trials
               if t["tid"] in _listify(tid)]
        df = pd.concat(res)

        # reorder columns
        fold_name = ["fold"] if "fold" in df else []
        df = _put_first(df, ["tid"] + fold_name + ["epoch"])
        return df

    def plot_history(self, tid, scores=["loss", "f1", "accuracy"],
                     figsize=(15, 3)):
        """Plot the loss curves"""
        history = self.train_history(tid)
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=figsize)
        for i, score in enumerate(scores):
            plt.subplot(1, len(scores), i + 1)
            plt.tight_layout()
            plt.plot(history[score], label="train")
            plt.plot(history['val_' + score], label="validation")
            plt.title(score)
            plt.ylabel(score)
            plt.xlabel('epoch')
            plt.legend(loc='best')
        return fig

    def load_model(self, tid, custom_objects=None):
        """Load saved keras model of the trial.

        If tid = None, get the best model

        Not applicable for trials ran in cross validion (i.e. not applicable
        for `CompileFN.cv_n_folds is None`
        """
        if tid is None:
            tid = self.best_trial_tid()

        model_path = self.get_trial(tid)["result"]["path"]["model"]
        return load_model(model_path, custom_objects=custom_objects)

    def n_ok(self):
        """Number of ok trials()
        """
        if len(self.trials) == 0:
            return 0
        else:
            return np.sum(np.array(self.statuses()) == "ok")

    def get_ok_results(self, verbose=True):
        """Return a list of results with ok status
        """
        if len(self.trials) == 0:
            return []

        not_ok = np.where(np.array(self.statuses()) != "ok")[0]

        if len(not_ok) > 0 and verbose:
            print("{0}/{1} trials were not ok.".format(len(not_ok), len(self.trials)))
            print("Trials: " + str(not_ok))
            print("Statuses: " + str(np.array(self.statuses())[not_ok]))

        r = [merge_dicts({"tid": t["tid"]}, t["result"].to_dict())
             for t in self.trials if t["result"]["status"] == "ok"]
        return r

    def as_df(self, ignore_vals=["history"], separator=".", verbose=True):
        """Return a pd.DataFrame view of the whole experiment
        """

        def add_eval(res):
            if "eval" not in res:
                if isinstance(res["history"], list):
                    # take the average across all folds
                    eval_names = list(res["history"][0]["loss"].keys())
                    eval_metrics = np.array([[v[-1] for k, v in hist["loss"].items()]
                                             for hist in res["history"]]).mean(axis=0).tolist()
                    res["eval"] = {eval_names[i]: eval_metrics[i] for i in range(len(eval_metrics))}
                else:
                    res["eval"] = {k: v[-1] for k, v in res["history"]["loss"].items()}
            return res

        def add_n_epoch(df):
            df_epoch = self.train_history().groupby("tid")["epoch"].max().reset_index()
            df_epoch.rename(columns={"epoch": "n_epoch"}, inplace=True)
            return pd.merge(df, df_epoch, on="tid", how="left")

        results = self.get_ok_results(verbose=verbose)
        rp = [_flatten_dict(_delete_keys(add_eval(x), ignore_vals), separator) for x in results]
        df = pd.DataFrame.from_records(rp)

        df = add_n_epoch(df)

        first = ["tid", "loss", "status"]
        return _put_first(df, first)


# --------------------------------------------
# TODO - put to a separate module
def _train_and_eval_single(train, valid, model,
                           batch_size=32, epochs=300, use_weight=False,
                           callbacks=[], eval_best=False, add_eval_metrics={}, custom_objects=None):
    """Fit and evaluate a keras model

    eval_best: if True, load the checkpointed model for evaluation
    """
    def _format_keras_history(history):
        """nicely format keras history
        """
        return {"params": history.params,
                "loss": merge_dicts({"epoch": history.epoch}, history.history),
                }
    if use_weight:
        sample_weight = train[2]
    else:
        sample_weight = None
    # train the model
    logger.info("Fit...")
    history = History()
    model.fit(train[0], train[1],
              batch_size=batch_size,
              validation_data=valid[:2],
              epochs=epochs,
              sample_weight=sample_weight,
              verbose=2,
              callbacks=[history] + callbacks)

    # get history
    hist = _format_keras_history(history)
    # load and eval the best model
    if eval_best:
        mcp = [x for x in callbacks if isinstance(x, ModelCheckpoint)]
        assert len(mcp) == 1
        model = load_model(mcp[0].filepath, custom_objects=custom_objects)

    return eval_model(model, valid, add_eval_metrics), hist


def eval_model(model, test, add_eval_metrics={}):
    """Evaluate model's performance on the test-set.

    # Arguments
        model: Keras model
        test: test-dataset. Tuple of inputs `x` and target `y` - `(x, y)`.
        add_eval_metrics: Additional evaluation metrics to use. Can be a dictionary or a list of functions
    accepting arguments: `y_true`, `y_predicted`. Alternatively, you can provide names of functions from
    the `kopt.eval_metrics` module.

    # Returns
        dictionary with evaluation metrics

    """
    # evaluate the model
    logger.info("Evaluate...")
    # - model_metrics
    model_metrics_values = model.evaluate(test[0], test[1], verbose=0,
                                          batch_size=test[1].shape[0])
    # evaluation is done in a single pass to have more precise metics
    model_metrics = dict(zip(_listify(model.metrics_names),
                             _listify(model_metrics_values)))
    # - eval_metrics
    y_true = test[1]
    y_pred = model.predict(test[0], verbose=0)
    eval_metrics = {k: v(y_true, y_pred) for k, v in add_eval_metrics.items()}

    # handle the case where the two metrics names intersect
    # - omit duplicates from eval_metrics
    intersected_keys = set(model_metrics).intersection(set(eval_metrics))
    if len(intersected_keys) > 0:
        logger.warning("Some metric names intersect: {0}. Ignoring the add_eval_metrics ones".
                       format(intersected_keys))
        eval_metrics = _delete_keys(eval_metrics, intersected_keys)

    return merge_dicts(model_metrics, eval_metrics)


def get_model(model_fn, train_data, param):
    """Feed model_fn with train_data and param
    """
    model_param = merge_dicts({"train_data": train_data}, param["model"], param.get("shared", {}))
    return model_fn(**model_param)


def get_data(data_fn, param):
    """Feed data_fn with param
    """
    return data_fn(**merge_dicts(param["data"], param.get("shared", {})))


class CompileFN():
    """Compile an objective function that

    - trains the model on the training set
    - evaluates the model on the validation set
    - reports the performance metric on the validation set as the objective loss

    # Arguments
        db_name: Database name of the KMongoTrials.
        exp_name: Experiment name of the KMongoTrials.
        data_fn: Tuple containing training data as the x,y pair at the first (index=0) element:
                 `((train_x, test_y), ...)`. If `valid_split` and `cv_n_folds` are both `None`,
                 the second (index=1) tuple is used as the validation dataset.
        add_eval_metrics: Additional list of (global) evaluation
            metrics. Individual elements can be
            a string (referring to kopt.eval_metrics)
            or a function taking two numpy arrays: `y_true`, `y_pred`.
            These metrics are ment to supplement those specified in
            `model.compile(.., metrics = .)`.
        optim_metric: str; Metric to optimize. Must be in
            `add_eval_metrics` or `model.metrics_names`.
        optim_metric_mode: one of {min, max}. In `min` mode,
            training will stop when the optimized metric
            monitored has stopped decreasing; in `max`
            mode it will stop when the optimized metric
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the optimized metric.
        valid_split: Fraction of the training points to use for the validation. If set to None,
                     the second element returned by data_fn is used as the validation dataset.
        cv_n_folds: If not None, use cross-validation with `cv_n_folds`-folds instead of train, validation split.
                    Overrides `valid_split` and `use_data_fn_valid`.
        stratified: boolean. If True, use stratified data splitting in train-validation split or cross-validation.
        random_state: Random seed for performing data-splits.
        use_tensorboard: If True, tensorboard callback is used. Each trial is written into a separate `log_dir`.
        save_model: It not None, the trained model is saved to the `save_dir` directory as hdf5 file.
                    If save_model="best", save the best model using `keras.callbacks.ModelCheckpoint`, and
                    if save_model="last", save the model after training it.
        save_results: If True, the return value is saved as .json to the `save_dir` directory.
        save_dir: Path to the save directory.
        custom_objects: argument passed to load_model - Optional dictionary mapping names (strings) to
             custom classes or functions to be considered during deserialization.
    """
    # TODO - check if we can get (db_name, exp_name) from hyperopt

    def __init__(self, db_name, exp_name,
                 data_fn,
                 model_fn,
                 # validation metric
                 add_eval_metrics=[],
                 optim_metric="loss",  # val_loss
                 optim_metric_mode="min",
                 # validation split
                 valid_split=.2,
                 cv_n_folds=None,
                 stratified=False,
                 random_state=None,
                 # saving
                 use_tensorboard=False,
                 save_model="best",
                 save_results=True,
                 save_dir=save_dir(),
                 custom_objects=None,
                 **kwargs
                 ):
        self.data_fn = data_fn
        self.model_fn = model_fn
        assert isinstance(add_eval_metrics, (list, tuple, set, dict))
        if isinstance(add_eval_metrics, dict):
            self.add_eval_metrics = {k: _get_ce_fun(v) for k, v in add_eval_metrics.items()}
        else:
            self.add_eval_metrics = {_to_string(fn_str): _get_ce_fun(fn_str)
                                     for fn_str in add_eval_metrics}
        assert isinstance(optim_metric, str)

        # backcompatibility:
        # allow only "loss_metric" and "loss_metric_mode" to be passed in kwargs
        if "loss_metric" in kwargs and optim_metric == "loss":
            optim_metric = kwargs["loss_metric"]
        if "loss_metric_mode" in kwargs and optim_metric_mode == "min":
            optim_metric_mode = kwargs["loss_metric_mode"]
        possible_kwargs = ["loss_metric", "loss_metric_mode"]
        add_arguments = set(kwargs.keys()).difference(possible_kwargs)

        if len(add_arguments) > 0:
            raise ValueError("Unknown argument(s) {0}. **kwargs accepts only arguments: {1}.  ".
                             format(add_arguments, possible_kwargs))

        self.optim_metric = optim_metric
        assert optim_metric_mode in ["min", "max"]
        self.optim_metric_mode = optim_metric_mode

        self.data_name = data_fn.__name__
        self.model_name = model_fn.__name__
        self.db_name = db_name
        self.exp_name = exp_name
        # validation
        self.valid_split = valid_split
        self.cv_n_folds = cv_n_folds
        self.stratified = stratified
        self.random_state = random_state
        # saving
        self.use_tensorboard = use_tensorboard
        self.save_dir = save_dir
        self.save_model = save_model if save_model is not None else ""
        self.save_results = save_results
        # loading
        self.custom_objects = custom_objects

        # backcompatibility
        if self.save_model is True:
            self.save_model = "last"
        elif self.save_model is False:
            self.save_model = ""
        assert self.save_model in ["", "last", "best"]

    @property
    def save_dir_exp(self):
        return self.save_dir + "/{db}/{exp}/".format(db=self.db_name, exp=self.exp_name)

    def _assert_optim_metric(self, model):
        model_metrics = _listify(model.metrics_names)
        eval_metrics = list(self.add_eval_metrics.keys())

        if self.optim_metric not in model_metrics + eval_metrics:
            raise ValueError("optim_metric: '{0}' not in ".format(self.optim_metric) +
                             "either sets of the losses: \n" +
                             "model.metrics_names: {0}\n".format(model_metrics) +
                             "add_eval_metrics: {0}".format(eval_metrics))

    def __call__(self, param):
        time_start = datetime.now()

        # set default early-stop parameters
        if param.get("fit") is None:
            param["fit"] = {}
        if param["fit"].get("epochs") is None:
            param["fit"]["epochs"] = 500
        # TODO - cleanup callback parameters
        #         - callbacks/early_stop/patience...
        if param["fit"].get("patience") is None:
            param["fit"]["patience"] = 10
        if param["fit"].get("batch_size") is None:
            param["fit"]["batch_size"] = 32
        if param["fit"].get("early_stop_monitor") is None:
            param["fit"]["early_stop_monitor"] = "val_loss"

        callbacks = [EarlyStopping(monitor=param["fit"]["early_stop_monitor"],
                                   patience=param["fit"]["patience"])]

        # setup paths for storing the data - TODO check if we can somehow get the id from hyperopt
        rid = str(uuid4())
        tm_dir = self.save_dir_exp + "/train_models/"
        if not os.path.exists(tm_dir):
            os.makedirs(tm_dir)
        model_path = tm_dir + "{0}.h5".format(rid) if self.save_model else ""
        results_path = tm_dir + "{0}.json".format(rid) if self.save_results else ""

        if self.use_tensorboard:
            max_len = 240 - len(rid) - 1
            param_string = _dict_to_filestring(_flatten_dict_ignore(param))[:max_len] + ";" + rid
            tb_dir = self.save_dir_exp + "/tensorboard/" + param_string[:240]
            callbacks += [TensorBoard(log_dir=tb_dir,
                                      histogram_freq=0,  # TODO - set to some number afterwards
                                      write_graph=False,
                                      write_images=True)]
        # -----------------

        # get data
        logger.info("Load data...")
        data = get_data(self.data_fn, param)
        train = data[0]
        if self.cv_n_folds is None and self.valid_split is None:
            valid_data = data[1]
        del data
        time_data_loaded = datetime.now()

        # train & evaluate the model
        if self.cv_n_folds is None:
            # no cross-validation
            model = get_model(self.model_fn, train, param)
            print(_listify(model.metrics_names))
            self._assert_optim_metric(model)
            if self.valid_split is not None:
                train_idx, valid_idx = split_train_test_idx(train,
                                                            self.valid_split,
                                                            self.stratified,
                                                            self.random_state)
                train_data = subset(train, train_idx)
                valid_data = subset(train, valid_idx)
            else:
                train_data = train

            c_callbacks = deepcopy(callbacks)
            if self.save_model == "best":
                c_callbacks += [ModelCheckpoint(model_path,
                                                monitor=param["fit"]["early_stop_monitor"],
                                                save_best_only=True)]
            eval_metrics, history = _train_and_eval_single(train=train_data,
                                                           valid=valid_data,
                                                           model=model,
                                                           epochs=param["fit"]["epochs"],
                                                           batch_size=param["fit"]["batch_size"],
                                                           use_weight=param["fit"].get("use_weight", False),
                                                           callbacks=c_callbacks,
                                                           eval_best=self.save_model == "best",
                                                           add_eval_metrics=self.add_eval_metrics,
                                                           custom_objects=self.custom_objects)
            if self.save_model == "last":
                model.save(model_path)
        else:
            # cross-validation
            eval_metrics_list = []
            history = []
            for i, (train_idx, valid_idx) in enumerate(split_KFold_idx(train,
                                                                       self.cv_n_folds,
                                                                       self.stratified,
                                                                       self.random_state)):
                logger.info("Fold {0}/{1}".format(i + 1, self.cv_n_folds))
                model = get_model(self.model_fn, subset(train, train_idx), param)
                self._assert_optim_metric(model)
                c_model_path = model_path.replace(".h5", "_fold_{0}.h5".format(i))
                c_callbacks = deepcopy(callbacks)
                if self.save_model == "best":
                    c_callbacks += [ModelCheckpoint(c_model_path,
                                                    monitor=param["fit"]["early_stop_monitor"],
                                                    save_best_only=True)]
                eval_m, history_elem = _train_and_eval_single(train=subset(train, train_idx),
                                                              valid=subset(train, valid_idx),
                                                              model=model,
                                                              epochs=param["fit"]["epochs"],
                                                              batch_size=param["fit"]["batch_size"],
                                                              use_weight=param["fit"].get("use_weight", False),
                                                              callbacks=c_callbacks,
                                                              eval_best=self.save_model == "best",
                                                              add_eval_metrics=self.add_eval_metrics,
                                                              custom_objects=self.custom_objects)
                print("\n")
                eval_metrics_list.append(eval_m)
                history.append(history_elem)
                if self.save_model == "last":
                    model.save(c_model_path)
            # summarize metrics - take average accross folds
            eval_metrics = _mean_dict(eval_metrics_list)

        # get loss from eval_metrics
        loss = eval_metrics[self.optim_metric]
        if self.optim_metric_mode == "max":
            loss = - loss  # loss should get minimized

        time_end = datetime.now()

        ret = {"loss": loss,
               "status": STATUS_OK,
               "eval": eval_metrics,
               # additional info
               "param": param,
               "path": {
                   "model": model_path,
                   "results": results_path,
               },
               "name": {
                   "data": self.data_name,
                   "model": self.model_name,
                   "optim_metric": self.optim_metric,
                   "optim_metric_mode": self.optim_metric,
               },
               "history": history,
               # execution times
               "time": {
                   "start": str(time_start),
                   "end": str(time_end),
                   "duration": {
                       "total": (time_end - time_start).total_seconds(),  # in seconds
                       "dataload": (time_data_loaded - time_start).total_seconds(),
                       "training": (time_end - time_data_loaded).total_seconds(),
                   }}}

        # optionally save information to disk
        if results_path:
            write_json(ret, results_path)
        logger.info("Done!")
        return ret

    # Style guide:
    # -------------
    #
    # path structure:
    # /s/project/deepcis/hyperopt/db/exp/...
    #                                   /train_models/
    #                                   /best_model.h5

    # hyper-params format:
    #
    # data: ... (pre-preprocessing parameters)
    # model: (architecture, etc)
    # train: (epochs, patience...)


# --------------------------------------------
# helper functions


def _delete_keys(dct, keys):
    """Returns a copy of dct without `keys` keys
    """
    c = deepcopy(dct)
    assert isinstance(keys, list)
    for k in keys:
        c.pop(k)
    return c


def _mean_dict(dict_list):
    """Compute the mean value across a list of dictionaries
    """
    return {k: np.array([d[k] for d in dict_list]).mean()
            for k in dict_list[0].keys()}


def _put_first(df, names):
    df = df.reindex(columns=names + [c for c in df.columns if c not in names])
    return df


def _listify(arg):
    if hasattr(type(arg), '__len__'):
        return arg
    return [arg, ]


def _get_ce_fun(fn_str):
    if isinstance(fn_str, str):
        return ce.get(fn_str)
    elif callable(fn_str):
        return fn_str
    else:
        raise ValueError("fn_str has to be callable or str")


def _flatten_dict(dd, separator='_', prefix=''):
    return {prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in _flatten_dict(vv, separator, kk).items()
            } if isinstance(dd, dict) else {prefix: dd}


def _flatten_dict_ignore(dd, prefix=''):
    return {k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in _flatten_dict_ignore(vv, kk).items()
            } if isinstance(dd, dict) else {prefix: dd}


def _dict_to_filestring(d):
    def to_str(v):
        if isinstance(v, float):
            return '%s' % float('%.2g' % v)
        else:
            return str(v)

    return ";".join([k + "=" + to_str(v) for k, v in d.items()])
