from hyperopt import fmin, tpe, hp, Trials
import os
import time

from kopt.hyopt import CompileFN, KMongoTrials
from kopt.hyopt import test_fn as fn_test
from kopt.utils import merge_dicts
import subprocess
from tests import data, model
from copy import deepcopy
import pytest


def test_argument_compileCN():
    a = CompileFN("test", "test2",
                  data_fn=data.data,
                  model_fn=model.build_model)

    # backcompatibility
    a = CompileFN("test", "test2",
                  data_fn=data.data,
                  model_fn=model.build_model,
                  optim_metric="acc",
                  optim_metric_mode="max")

    a = CompileFN("test", "test2",
                  data_fn=data.data,
                  model_fn=model.build_model,
                  loss_metric="acc",
                  loss_metric_mode="max")

    # raises error
    with pytest.raises(ValueError) as excinfo:
        a = CompileFN("test", "test2",
                      data_fn=data.data,
                      model_fn=model.build_model,
                      loss_metric="acc",
                      loss_metric_mode="max",
                      unknown_arg=3)


def test_compilefn_train_test_split(tmpdir):
    db_name = "test"
    exp_name = "test2"
    fn = CompileFN(db_name, exp_name,
                   data_fn=data.data,
                   model_fn=model.build_model,
                   optim_metric="acc",
                   optim_metric_mode="max",
                   # eval
                   valid_split=.5,
                   stratified=False,
                   random_state=True,
                   save_dir="/tmp/")
    hyper_params = {
        "data": {},
        "shared": {"max_features": 100, "maxlen": 20},
        "model": {"filters": hp.choice("m_filters", (2, 5)),
                  "hidden_dims": 3,
                  },
        "fit": {"epochs": 1}
    }
    fn_test(fn, hyper_params, tmp_dir=str(tmpdir))
    trials = Trials()
    best = fmin(fn, hyper_params, trials=trials, algo=tpe.suggest, max_evals=2)
    assert isinstance(best, dict)


def test_compilefn_cross_val(tmpdir):
    db_name = "test"
    exp_name = "test2"
    fn = CompileFN(db_name, exp_name,
                   cv_n_folds=3,
                   stratified=False,
                   random_state=True,
                   data_fn=data.data,
                   model_fn=model.build_model,
                   optim_metric="loss",
                   optim_metric_mode="min",
                   save_dir="/tmp/")
    hyper_params = {
        "data": {},
        "shared": {"max_features": 100, "maxlen": 20},
        "model": {"filters": hp.choice("m_filters", (2, 5)),
                  "hidden_dims": 3,
                  },
        "fit": {"epochs": 1}
    }
    fn_test(fn, hyper_params, tmp_dir=str(tmpdir))
    trials = Trials()
    best = fmin(fn, hyper_params, trials=trials, algo=tpe.suggest, max_evals=2)
    assert isinstance(best, dict)


def test_hyopt(tmpdir):
    # get the base dir
    mongodb_path = str(tmpdir.mkdir('mongodb'))
    results_path = str(tmpdir.mkdir('results'))
    # mongodb_path = "/tmp/mongodb_test/"
    # results_path = "/tmp/results/"

    proc_args = ["mongod",
                 "--dbpath=%s" % mongodb_path,
                 "--noprealloc",
                 "--port=22334"]
    print("starting mongod", proc_args)
    mongodb_proc = subprocess.Popen(
        proc_args,
        # stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
        cwd=mongodb_path,  # this prevented mongod assertion fail
    )

    # wait a bit
    time.sleep(1)
    proc_args_worker = ["hyperopt-mongo-worker",
                        "--mongo=localhost:22334/test",
                        "--poll-interval=0.1"]

    mongo_worker_proc = subprocess.Popen(
        proc_args_worker,
        # stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
        env=merge_dicts(os.environ, {"PYTHONPATH": os.getcwd()}),
    )
    # --------------------------------------------

    db_name = "test"
    exp_name = "test2"

    fn = CompileFN(db_name, exp_name,
                   data_fn=data.data,
                   model_fn=model.build_model,
                   optim_metric="acc",
                   optim_metric_mode="max",
                   save_dir=results_path)
    hyper_params = {
        "data": {},
        "shared": {"max_features": 100, "maxlen": 20},
        "model": {"filters": hp.choice("m_filters", (2, 5)),
                  "hidden_dims": 3,
                  },
        "fit": {"epochs": 1}
    }
    fn_test(fn, hyper_params, tmp_dir=str(tmpdir))
    trials = KMongoTrials(db_name, exp_name, ip="localhost",
                          kill_timeout=5 * 60,
                          port=22334)

    best = fmin(fn, hyper_params, trials=trials, algo=tpe.suggest, max_evals=2)
    assert len(trials) == 2
    assert len(trials) == trials.n_ok()
    assert isinstance(best, dict)
    assert "m_filters" in best

    # test my custom functions
    trials.as_df()
    trials.train_history(trials.valid_tid()[0])
    trials.train_history(trials.valid_tid())
    trials.get_ok_results()
    tid_best = trials.best_trial_tid()
    assert tid_best == trials.best_trial["tid"]
    assert trials.optimal_epochs(tid_best) == 1

    # --------------------------------------------
    # cross-validation
    db_name = "test"
    exp_name = "test2_cv"

    fn = CompileFN(db_name, exp_name,
                   data_fn=data.data,
                   model_fn=model.build_model,
                   cv_n_folds=3,
                   save_dir=results_path)

    trials = KMongoTrials(db_name, exp_name, ip="localhost",
                          kill_timeout=5 * 60,
                          port=22334)
    fn_test(fn, hyper_params, tmp_dir=str(tmpdir))
    best = fmin(fn, deepcopy(hyper_params), trials=trials, algo=tpe.suggest, max_evals=2)
    assert len(trials) == 2
    assert len(trials) == trials.n_ok()
    assert isinstance(best, dict)
    assert "m_filters" in best

    # test my custom functions
    trials.as_df()
    trials.train_history(trials.valid_tid()[0])
    trials.train_history(trials.valid_tid())
    trials.get_ok_results()
    tid_best = trials.best_trial_tid()
    assert tid_best == trials.best_trial["tid"]
    assert trials.optimal_epochs(tid_best) == 1

    assert trials.best_trial_tid() == trials.best_trial["tid"]
    # --------------------------------------------
    # close
    mongo_worker_proc.terminate()
    mongodb_proc.terminate()
