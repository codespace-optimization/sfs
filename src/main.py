import hydra
from omegaconf import DictConfig, OmegaConf
# from src.experiment_manager import CodingExperimentManager
from src.experiment_manager_plus import CodingExperimentManager
from importlib import import_module
import importlib
import os
import logging

original_cwd = os.getcwd()

def get_real_path(path: str) -> str:
    '''
    Gets the real, non-hydra modified path using the original cwd
    '''
    return os.path.join(original_cwd, path)

def import_class(relative_module_path: str, class_name: str):
    try:
        module = importlib.import_module(relative_module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Error importing {class_name} from {relative_module_path}: {str(e)}")

@hydra.main(config_path='conf', config_name='config', version_base='1.2', )
def main(cfg: DictConfig) -> None:
    logging.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    original_solution_set_path = get_real_path(cfg.paths.solution_set_path)
    logging.info(f"Original solution set path: {original_solution_set_path}")
    # Ensure the solution directory exists
    os.makedirs(os.path.dirname(original_solution_set_path), exist_ok=True)

    original_report_card_path = get_real_path(cfg.paths.report_card_path)
    logging.info(f"Original report card path: {original_report_card_path}")
    # Ensure the report card directory exists
    os.makedirs(os.path.dirname(original_report_card_path), exist_ok=True)

    original_results_path = get_real_path(cfg.paths.results_path)
    logging.info(f"Original results path: {original_results_path}")
    # Ensure the report card directory exists
    os.makedirs(os.path.dirname(original_results_path), exist_ok=True)

    # Dynamic import of the solver class
    solver = import_class(cfg.solver.class_module, cfg.solver.class_name)(**cfg.solver.params)

    # Dynamic import of the executor class
    executor = import_class(cfg.executor.class_module, cfg.executor.class_name)(**cfg.executor.params)

    dataset_name = cfg.dataset_name
    # if dataset_name ends with _plus, then it is in plus format
    is_plus_format = dataset_name.endswith("_plus")
    if is_plus_format:
        problem_set_name = dataset_name
        problem_set_path = None
    else:
        problem_set_path = cfg.paths.problem_set_path
        problem_set_name = None

    # # if the file in `problem_set_path` starts with `plus_`, then the problem set is in plus format
    # is_plus_format = os.path.basename(cfg.paths.problem_set_path).startswith("plus_")

    # Initialize the CodingProblemManager with paths from the configuration
    manager = CodingExperimentManager(
        solver=solver,
        executor=executor,
        problem_set_path=problem_set_path,
        test_set_path=cfg.paths.test_set_path,
        solution_set_path=original_solution_set_path,
        report_card_path=original_report_card_path,
        results_path=original_results_path,
        is_plus_format=is_plus_format,
        problem_set_name=problem_set_name
    )

    manager.append_to_report_card(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Example operation: solve unsolved problems
    manager.solve_unsolved_problems()

if __name__ == "__main__":
    main()