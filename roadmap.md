## Roadmap

- move model training, evaluation etc into a separate file 
  - need to modularize this functionality
    - maybe define a model abstract class?
  - also, specify custom callbacks etc
- host the documentation
  - setup readthedocs webpage?
- support data generators for the data() function
- feat: callbacks.EvalMetrics
  - use the whole validation set to compute the validation auprc at the end of the epoch
- feat: save CompileFN to MongoDB or disk. 
  - That way, the only information we would need would be the db_name and exp_name in order to retrain the model.
- feat: implement KMongoTrials.delete_submitted
  - delete jobs that were submitted but not pulled from the worker
