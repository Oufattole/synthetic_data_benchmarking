from pycaret.classification import * # Preprocessing, modelling, interpretation, deployment...
import pandas as pd # Basic data manipulation
from sklearn.model_selection import train_test_split # Data split
from sdv.tabular import CopulaGAN, GaussianCopula, CTGAN, TVAE # Synthetic data
from sdv.evaluation import evaluate # Evaluate synthetic data
import sdmetrics
import sklearn
import os
import task
import pickle

SDV_GENGERATORS = ["ct_gan", "gaussain_copula", "copula_gan", "tvae"]
CLASSIFICATION_METRICS = ['AUC', 'F1', 'Recall', 'Precision', 'Accuracy']

def split_data(dataset_path ="data.csv", output_directory="data", train_filename = "train.csv", test_filename = "test.csv", target_name="TARGET"):
    """Split real data into train and test set
    Args:
        dataset_path:
            path to dataset csv
        output_directory:
            relative or absolute directory to store outputs in
        train_filename:
            filename to use for train data split, include .csv
        test_filename:
            filename to use for test data split, include .csv
        target_name:    name of target column in dataset
    Return:
        list:
            [train_dataset output path, test_dataset output path] 
    """
    data = pd.read_csv(dataset_path)
    data.head()
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    train, test, target_train, target_test = train_test_split(data.drop(target_name, axis = 1), data[target_name], test_size = 0.4, random_state = 42)
    train[target_name] = target_train
    test[target_name] = target_test
    train.to_csv(os.path.join(output_directory, train_filename))
    test.to_csv(os.path.join(output_directory, test_filename))


def create_default_generators(train_dataset="data/train.csv", generators=SDV_GENGERATORS, output_directory="generators"):
    """Create SDV Generators with default hyperparameters
    Args:
        train_dataset:
            the path to train dataset
        generators:
            the sdv generators to use, must be a subset of SDV_GENGERATORS
        output_directory:
            the directory to store generator pickle files
    Returns:
        list:
            a list of output_paths of the generator pickle files
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    data = pd.read_csv(train_dataset)
    models = []
    name_to_model = \
    {"ct_gan":CTGAN, "gaussain_copula":GaussianCopula, "copula_gan":CopulaGAN, "tvae":TVAE}
    for name in generators:
        assert(name in SDV_GENGERATORS)
        models.append(name_to_model[name])
    trained_models = []
    output_paths = []
    for model, name in zip(models, generators):
        model_instance = model()
        model_instance.fit(data)
        output_path = os.path.join(output_directory, "default_" + name + ".pkl")
        model_instance.save(output_path)
        output_paths.append(output_path)
    return output_paths

    def benchmark(tasks, metrics=CLASSIFICATION_METRICS, agnostic_metrics=False,
                output_path='results/', save_results=True):
  """Run benchmark testing on a set of tasks. Return detailed results of each run stored in a
    DataFrame object.
    Args:
        tasks (list):
            a list of task instances storing meta information of each task.
        metrics (list):
            pycaret classification metrics to record
            a subset of ['Accuracy', 'AUC', 'Recall', 'F1', 'Precision', 'Kappa']
        agnostic_metrics (boolean):
            whether to record dataset agnostic metrics in results
        output_path (str):
            the dir path to store benchmark results and records of each task.
        save_results (boolean):
            whether to store the benchmark results.
        TODO add option to store generated synthetic data
        TODO add regression benchmarking
    Returns:
        pd.DataFrame:
            benchmarking results in detail.
    """
    if output_path is not None:
        if not os.path.exists(output_path):
            os.mkdir(output_path)

    performance = []
    for task in tasks:
        if output_path is not None:
            task_output_path = os.path.join(output_path, task.task_id)
        else:
            task_output_path = None
        performance.extend(evaluate_task(task=task, metrics=metrics, output_path=task_output_path,
                                         agnostic_metrics=agnostic_metrics))
    result_df = pd.DataFrame.from_records(performance)

    if output_path is not None and save_results:
        result_df.to_csv(os.path.join(output_path, 'details.csv'))

    return result_df


def evaluate_task(task, metrics, output_path, agnostic_metrics):
    """Run benchmark testing on a task. Save intermedia data, trained models, and optimized
    hyperparameters. Return testing results.

    Args:
        task (Task):
            a task instance storing meta information of the task.
        metrics (list)
            a list of strings to identify the metric functions.
        output_path (str):
            a directory path to store the intermedia data, model and hyperparametes.
        agnostic_metrics (boolean):
            whether to record dataset agnostic metrics in results

    Returns:
        list:
            benchmarking results of each run.
    """
    # Load Data
    train_data = pd.read_csv(task.train_data)
    test_data = pd.read_csv(test.train_data)

    # Run the task for #task.run_num times and record each run.
    results = []
    task_id,target,pycaret_model,run_num
    generator = pickle.load(task.path_to_generator)
    for train_dataset, agnostic_scores in _prepare_training_datasets(task.sampling_method, generator, task.run_num):
        scores = _evaluate_synthetic_data(training_dataset, test_dataset, metrics, agnostic_metrics)
        #TODO add ability to store models and generated data
        results.append(scores)
        
    # Store the task specific results.
    if output_path is not None:
        # Initialize the output directory.
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)
        #TODO store tasks metadata in results

    return results

def augmented_training_datasets(train_data, target, generator, sampling_method, run_num):
    """ yields datasets augmented with sampled data
    Yields:
        dataset, agnostic_scores, sampling_method_run_id
    """
    i = 0
    #baseline
    yield train_data, None,"baseline_" + str(i)
    
    if sampling_method == "original" or sampling_method == "all":
        
        num_train_samples = train_data.shape[0]//
        step_size = num_train_samples // 2
        for sample_size in range(1, num_train_samples, step_size):
            synthetic_data = generator.sample(sample_size)
            score_aggregate = evaluate(synthetic_data, train, aggregate=False)
            score_column = make_score_column(score_aggregate)
            combined_data = train.append(synthetic_data)
            yield 
            i+=1
    
    if sampling_method == "uniform" or sampling_method == "all":
        target_category_frequencies = train_data['status'].value_counts().to_dict()
        largest_target =train_data[target].value_counts().nlargest(n=1).values[0] # get largest target category
        for each in target_category_frequency:
            #do rejection sampling
        
        num_target_samples = train_data[train_data[target] == ]
        synthetic_data = generator.sample(sample_size)
        score_aggregate = evaluate(synthetic_data, train, aggregate=False)
        score_column = make_score_column(score_aggregate)
        combined_data = train.append(synthetic_data)
        yield 
        i+=1

    
    if task.sampling_method ==  "all":
        yield from _sample_all(generator)
    elif task.sampling_method ==  "original":
        yield from _sample_original(generator)
    elif task.sampling_method ==  "uniform":
        yield from _sample_uniform(generator)
    else:
        raise ValueError("for task id {} task.sampling_method is {} which is invalid".format(task.task_id, task.sampling_method))
#########################################################################################################
results = pd.DataFrame(columns=columns)
count = 0
for model in models:
    model_column = [str(model)]
    for sample_size in sample_sizes:
        sample_size_column = [sample_size]
        print(sample_size)
        print(model)
        synthetic_data = model.sample(int(sample_size))
        score_aggregate = evaluate(synthetic_data, train, aggregate=False)
        score_column = make_score_column(score_aggregate)
        combined_data = train.copy().append(synthetic_data)
        ord_feats = {}
        for feat in ord_levels:
            ord_feats[feat] = [str(each) for each in sorted(list(combined_data[feat].unique()))]
        classifier_setup(combined_data)
        for classifier_name in classifiers_test:
            classifier_name = classifiers[0]
            classifier_column = [classifier_name]
            classifier = create_model(classifier_name)
            #pycaret, predict on test set
            pred_holdout = predict_model(classifier, verbose=False)
            #evaluate performance
            y_true = pred_holdout["Attrition"]
            y_pred = pred_holdout["Label"]
            prf = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred)
            precision = prf[0][1]
            recall = prf[1][1]
            f1 = prf[2][1]
            accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
            performance_column = [accuracy, f1, recall, precision]
            #record entry
            results_row = model_column + sample_size_column + score_column + classifier_column + performance_column
            row_dict = {columns[i]:results_row[i] for i in range(len(columns))}
            results = results.append(row_dict, ignore_index=True)
            print(results.shape)
            count+=1
            print(count)
#########################################################################################################

def _evaluate_synthetic_data(run_id, pipeline, feature_matrix, pipeline_name=None, problem_name=None,
                       dataset_name=None, beginning_stage=None, optimize=False, metrics=None):
    """Evaluate a pipeline's performance on a target dataset according to the given metrics.

    Args:
        run_id (int):
            the index to specify the execution to the pipeline.
        pipeline (MLPipeline):
            a pipeline instance.
        feature_matrix (pd.DataFrame):
            a dataframe consists of both feature values and target values.
        pipeline_name (str):
            the name of the pipeline.
        problem_name (str):
            the name of the problem.
        dataset_name (str):
            the name of the dataset.
        beginning_stage (str):
            the stage in which the benchmarking are applied, should be either "data_loader",
            "problem_definition", "featurization".
        optimize (boolean):
            whether to optimize the hyper-parameters of the pipeline.
        metrics (list)
            a list of strings to identify the metric functions.

    Returns:
        tuple:
            pipeline evaluation results including (performance, models, hyperparameters).
    """
    modeler = Modeler(pipeline, PROBLEM_TYPE[problem_name])

    features, target = _split_features_target(feature_matrix, problem_name)
    # TODO: digitize the labels in the featurization (problem definition) stage.
    if problem_name == 'LOS' and dataset_name == 'mimic-iii':
        target = np.digitize(target, [0, 7])

    LOGGER.info("Starting pipeline {} for {} problem..".format(pipeline_name, problem_name))

    start = datetime.utcnow()
    try:
        scores = modeler.evaluate(features, target,
                                  tune=optimize, scoring='F1 Macro', metrics=metrics, max_evals=10)
        elapsed = datetime.utcnow() - start
        model = modeler.pipeline
        hyperparameters = modeler.pipeline.get_hyperparameters() if optimize else None
        scores['Elapsed Time(s)'] = elapsed.total_seconds()
        scores['Status'] = 'OK'

    except Exception as ex:
        LOGGER.exception(
            "Exception scoring pipeline {} in problem {}, exception {}".format(pipeline_name,
                                                                               problem_name, ex))
        elapsed = datetime.utcnow() - start
        model = None
        hyperparameters = None
        scores = {'Elapsed Time(s)': elapsed.total_seconds(), 'Status': 'Fail'}

    scores['Pipeline Name'] = pipeline_name
    scores['Run #'] = run_id
    scores['Problem Name'] = problem_name
    scores['Dataset Name'] = dataset_name
    scores['Beginning Stage'] = beginning_stage
    scores['Tuned'] = optimize

    return scores, model, hyperparameters