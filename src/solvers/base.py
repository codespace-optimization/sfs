import enum

from sympy import true
from ..headers import CodingProblem, CodingProblemAndSolution, Solver
from ..generators.generator_types import Generator, ModelBase
from ..generators.generator import GenericChatGenerator
from ..generators.factory import generator_factory, model_factory
from ..executors.executor_types import Executor
from ..executors.factory import executor_factory
from .preambles import PREAMBLES_JABBERWOCKY, PREAMBLES_ROLE_GPT_GENERATED, PREAMBLES_INSTRUCTION_GPT_GENERATED


from typing import Any, Optional
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher
import ast
import networkx as nx
import tokenize
import io



class BaseSolver(Solver):
    '''
    Uses reflexion (feedback from executing tests) to iteratively improve the solution
    '''

    def __init__(self, generator: GenericChatGenerator | str, model: ModelBase | str, executor: Executor | str, number_of_tests: int = 6, temperature: float = 0.6, num_seeds: int = 1, preamble_mode: str = "none", testgen_method: str = "simple", filter_num_tests: int = 100, filter_weighting_mode: str = 'uniform', do_fill_with_tests: bool = True, rng: np.random.Generator = np.random.default_rng(42)):
        super().__init__()
        if isinstance(generator, str):
            generator = generator_factory(generator)
        if isinstance(model, str):
            model = model_factory(model)
        if isinstance(executor, str):
            executor = executor_factory(executor)
        self.generator = generator
        self.model = model
        self.executor = executor
        self.number_of_tests = number_of_tests
        self.num_seeds = num_seeds
        self.preamble_mode = preamble_mode
        self.testgen_method = testgen_method
        self.temperature = temperature
        self.filter_num_tests = filter_num_tests
        self.filter_weighting_mode = filter_weighting_mode
        self.do_fill_with_tests = do_fill_with_tests
        self.rng = rng

    @staticmethod
    def cosine_similarity(code_snippets: list[str]) -> float:
        """
        Computes the average semantic similarity between a list of code snippets.

        Args:
        - code_snippets: List of code snippets as strings.

        Returns:
        - A float representing the average pairwise semantic similarity score.
        """
        # Load a pretrained model and tokenizer for code understanding (e.g., CodeBERT)
        model_name = "microsoft/codebert-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Get embeddings for each code snippet
        def get_code_embedding(code: str):
            inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            # Use the embedding of the [CLS] token as the sentence-level embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            # Normalize the embedding
            return embedding / np.linalg.norm(embedding)

        # Get embeddings for all code snippets
        embeddings = np.vstack([get_code_embedding(snippet) for snippet in code_snippets])

        # Calculate pairwise cosine similarities
        similarities = cosine_similarity(embeddings)

        # Average the pairwise similarities, excluding self-similarities (the diagonal)
        n = len(code_snippets)
        if n < 2:
            return 1.0  # If there's only one snippet, similarity is trivially 1

        total_sim = (np.sum(similarities) - np.trace(similarities)) / (n * (n - 1))

        return total_sim

    @staticmethod
    def tfidf_similarity(code_snippets: list[str]) -> float:
        """
        Calculate similarity using TF-IDF vectorization.
        """
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(code_snippets)
        similarities = cosine_similarity(tfidf_matrix)
        return float(np.mean(similarities[np.triu_indices(len(code_snippets), k=1)]))
    
    @staticmethod
    def levenshtein_similarity(code_snippets: list[str]) -> float:
        """
        Calculate similarity using Levenshtein distance.
        """
        def levenshtein(s1, s2):
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]

        max_length = max(len(s) for s in code_snippets)
        similarities = []
        for i in range(len(code_snippets)):
            for j in range(i+1, len(code_snippets)):
                distance = levenshtein(code_snippets[i], code_snippets[j])
                similarity = 1 - (distance / max_length)
                similarities.append(similarity)
        return float(np.mean(similarities))

    @staticmethod
    def token_sequence_similarity(code_snippets: list[str]) -> float:
        """
        Calculate similarity based on token sequence matching.
        """
        def tokenize_code(code):
            try:
                tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
                return [token.string for token in tokens if token.string.strip()]
            except tokenize.TokenError:
                # Handle incomplete or invalid code
                return code.split()

        similarities = []
        for i in range(len(code_snippets)):
            for j in range(i+1, len(code_snippets)):
                tokens1 = tokenize_code(code_snippets[i])
                tokens2 = tokenize_code(code_snippets[j])
                matcher = SequenceMatcher(None, tokens1, tokens2)
                similarity = matcher.ratio()
                similarities.append(similarity)
        return float(np.mean(similarities))

    @staticmethod
    def ast_similarity(code_snippets: list[str]) -> float:
        """
        Calculate similarity based on Abstract Syntax Tree (AST) structure.
        """
        def ast_to_graph(code):
            tree = ast.parse(code)
            graph = nx.Graph()
            for node in ast.walk(tree):
                graph.add_node(id(node), type=type(node).__name__)
                for child in ast.iter_child_nodes(node):
                    graph.add_edge(id(node), id(child))
            return graph

        def graph_edit_distance(g1, g2):
            return nx.graph_edit_distance(g1, g2, node_match=lambda n1, n2: n1['type'] == n2['type'])

        graphs = [ast_to_graph(snippet) for snippet in code_snippets]
        max_nodes = max(len(g.nodes) for g in graphs)
        similarities = []
        for i in range(len(graphs)):
            for j in range(i+1, len(graphs)):
                distance = graph_edit_distance(graphs[i], graphs[j])
                similarity = 1 - (distance / max_nodes)
                similarities.append(similarity)
        return float(np.mean(similarities))



    def filter(self, solutions: list[CodingProblemAndSolution], weighting_mode: str = "uniform") -> CodingProblemAndSolution:
        '''
        Filters the solution for the best one
        '''
        generated_tests = []
        generated_tests_pass_count = np.zeros(self.filter_num_tests)
        for i in range(self.filter_num_tests):
            generated_tests += self.generator.generate_internal_tests(func_sig=solutions[0].prompt, model=self.model, max_num_tests=self.number_of_tests, entry_point=solutions[0].entry_point, temperature=self.temperature, method=self.testgen_method)
        generated_tests = generated_tests[:self.filter_num_tests]
        assert len(generated_tests) == self.filter_num_tests, f"len(generated_tests)={len(generated_tests)} != {self.filter_num_tests}"

        for solution in solutions:
            solution.visible_tests = tuple(generated_tests)
            _, _, test_results = self.executor.execute(solution, will_modify=False)

            # add test results to generated_tests_pass_count
            generated_tests_pass_count += np.array(test_results)
            solution.extra_unwritten_kwargs['final_test_results'] = np.array(test_results)
            
        for solution in solutions:
            # take dot product of test_results and generated_tests_pass_count to get weighted score
            solution.extra_kwargs['final_weighted_score'] = float(np.dot(solution.extra_unwritten_kwargs['final_test_results'], generated_tests_pass_count)/len(generated_tests))
            solution.extra_kwargs['final_score'] = float(sum(solution.extra_unwritten_kwargs['final_test_results'])/len(generated_tests))
        
        if weighting_mode == "uniform":
            best_solution = max(solutions, key=lambda x: x.extra_kwargs['final_score'])
        elif weighting_mode == "passing_solutions":
            best_solution = max(solutions, key=lambda x: x.extra_kwargs['final_weighted_score'])
        else:
            raise ValueError(f"Invalid weighting mode {weighting_mode}")
        # raise ValueError(f"Tested {len(generated_tests)} tests, best score: {best_score}")
        return best_solution

    def fill_with_tests(self, solution: CodingProblem, num_tests: int, max_tries: int = 20) -> None:
        '''
        Fill the solution with tests until there are exactly num_tests
        '''
        if self.do_fill_with_tests:
            # generate internal tests if they are less than the number of tests
            counter = 0
            while len(solution.visible_tests) < num_tests:
                solution.visible_tests = tuple(solution.visible_tests) + tuple(self.generator.generate_internal_tests(func_sig=solution.prompt, model=self.model, max_num_tests=self.number_of_tests, entry_point=solution.entry_point, temperature=self.temperature, method=self.testgen_method))
                counter += 1
                if counter > max_tries:
                    raise ValueError("Cannot generate enough internal tests")
            solution.visible_tests = solution.visible_tests[:num_tests]

    def is_any_solved(self, solutions: list[CodingProblemAndSolution]) -> tuple[Optional[CodingProblemAndSolution], bool, Optional[int]]:
        '''
        Check if any of the solutions are solved

        Returns:
            solution: The first solution that is solved 
            is_solved: whether any solution is solved
            index: the index of the solution that is solved
        '''
        for i, solution in enumerate(solutions):
            if self.executor.evaluate(solution, will_modify=False):
                return solution, True, i
        return None, False, None
    
    # @staticmethod
    # def calculate_average_test_score(solutions: list[CodingProblemAndSolution]) -> float:
    #     '''
    #     Calculate the average test score of the solutions
    #     '''
    #     total_score = 0
    #     for solution in solutions:
    #         total_score += sum(solution.test_results)
    #     return total_score/len(solutions)
    
    def generate_seed_functions(self, num_seeds: int, problem: CodingProblem, stop_if_solved: bool = False) -> list[CodingProblemAndSolution]:
        '''
        Generate seed functions
        '''
        seed_functions: list[CodingProblemAndSolution] = []
        assert num_seeds >= 1
        # iteratively add seed functions
        for i in range(num_seeds):
            cur_solution = CodingProblemAndSolution.init_from_coding_problem(problem)

            if self.preamble_mode == "role":
                shuffled_preambles = self.rng.permutation(PREAMBLES_ROLE_GPT_GENERATED)
                preamble = shuffled_preambles[i % len(PREAMBLES_ROLE_GPT_GENERATED)]
            elif self.preamble_mode == "none":
                preamble = ''
            elif self.preamble_mode == "instruction":
                shuffled_preambles = self.rng.permutation(PREAMBLES_INSTRUCTION_GPT_GENERATED)
                preamble = shuffled_preambles[i % len(PREAMBLES_INSTRUCTION_GPT_GENERATED)]
            elif self.preamble_mode == "jabberwocky":
                shuffled_preambles = self.rng.permutation(PREAMBLES_JABBERWOCKY)
                preamble = shuffled_preambles[i % len(PREAMBLES_JABBERWOCKY)]
            else:
                raise ValueError(f"Invalid preamble mode {self.preamble_mode}")

            # first attempt
            counter = 0
            while cur_solution.solution is None:
                first_strategy =  self.generator.generate_func_impl(func_sig=problem.prompt, model=self.model, strategy="simple", temperature=self.temperature, preamble=preamble, entry_point=problem.entry_point)[0]
                cur_solution.solution = first_strategy.code
                cur_solution.extra_kwargs['logprob'] = first_strategy.logprob
                counter += 1
                if counter > 10:
                    raise ValueError("Cannot generate a function implementation")

            # record token cost
            cur_solution.extra_kwargs['int_output_tokens'] = self.model.output_tokens
            cur_solution.extra_kwargs['int_total_tokens'] = self.model.total_tokens
            cur_solution.extra_kwargs['int_total_requests'] = self.model.total_requests

            assert isinstance(cur_solution.solution, str)
            seed_functions.append(cur_solution)

            # execute the solution
            self.executor.execute(cur_solution, will_modify=True)

            # break if the solution is solved
            if all(cur_solution.test_results) and len(cur_solution.visible_tests) > 0:
                break

            if stop_if_solved and self.executor.evaluate(cur_solution, will_modify=False):
                break

        return seed_functions
    
    def record_sample_stats_on_solution(self, all_solutions: list[CodingProblemAndSolution], best_solution: CodingProblemAndSolution) -> None:
        '''
        Record statistics about the solutions in the submitted solution
        '''
        solved_solution, any_solved, solved_index = self.is_any_solved(all_solutions)

        best_solution.extra_kwargs['any_solved'] = any_solved
        best_solution.extra_kwargs['solved_score'] = sum(solved_solution.test_results)/len(solved_solution.test_results) if solved_solution is not None and len(solved_solution.test_results) != 0 else None
        best_solution.extra_kwargs['solved_index'] = solved_index
        best_solution.extra_kwargs['solved_output_tokens'] = solved_solution.extra_kwargs['int_output_tokens'] if solved_solution is not None else None
        best_solution.extra_kwargs['solved_total_tokens'] = solved_solution.extra_kwargs['int_total_tokens'] if solved_solution is not None else None
        best_solution.extra_kwargs['solved_total_requests'] = solved_solution.extra_kwargs['int_total_requests'] if solved_solution is not None else None


        best_solution.extra_kwargs['output_tokens'] = self.model.output_tokens
        best_solution.extra_kwargs['total_tokens'] = self.model.total_tokens
        best_solution.extra_kwargs['total_requests'] = self.model.total_requests

        proposed_solutions = [solution.solution for solution in all_solutions if solution.solution is not None]
        best_solution.extra_kwargs['cosine_similarity'] = self.cosine_similarity(proposed_solutions)
        best_solution.extra_kwargs['tfidf_similarity'] = self.tfidf_similarity(proposed_solutions)
        best_solution.extra_kwargs['levenshtein_similarity'] = self.levenshtein_similarity(proposed_solutions)
        # best_solution.extra_kwargs['ast_similarity'] = self.ast_similarity(proposed_solutions)
        best_solution.extra_kwargs['token_sequence_similarity'] = self.token_sequence_similarity(proposed_solutions)
        best_solution.extra_kwargs['num_solutions'] = len(all_solutions)
        best_solution.extra_kwargs['avg_solved'] = sum(solution.is_solved for solution in all_solutions)/len(all_solutions)
        best_solution.extra_kwargs['scores'] = [sum(all_solutions[i].test_results)/len(all_solutions[i].test_results) if len(all_solutions[i].test_results) != 0 else 0.0 for i in range(len(all_solutions))]
        best_solution.extra_kwargs['avg_score'] = sum(best_solution.extra_kwargs['scores'])/len(best_solution.extra_kwargs['scores'])
        
        