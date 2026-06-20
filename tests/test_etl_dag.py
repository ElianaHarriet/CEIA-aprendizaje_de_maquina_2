"""Structural tests for etl_movielens DAG using AST analysis.

Airflow has compatibility issues with Python 3.14, so we parse the DAG
file with the `ast` module to verify structure without importing it.
"""

import ast
from pathlib import Path

import pytest

DAG_PATH = Path(__file__).resolve().parent.parent / "airflow" / "dags" / "etl_movielens.py"


class TestETLDagStructure:
    """Verify etl_movielens.py defines the expected pipeline structure."""

    @classmethod
    def setup_class(cls):
        with open(DAG_PATH) as f:
            cls.tree = ast.parse(f.read())
        cls.function_defs = {
            node.name: node
            for node in ast.walk(cls.tree)
            if isinstance(node, ast.FunctionDef)
        }

    def test_dag_function_exists(self):
        assert "etl_movielens" in self.function_defs

    def test_dag_has_expected_tasks(self):
        dag_func = self.function_defs["etl_movielens"]
        task_names = [
            node.name
            for node in dag_func.body
            if isinstance(node, ast.FunctionDef)
        ]
        expected_tasks = [
            "download_data",
            "sample_and_save_ratings",
            "compute_movie_features",
            "compute_genome_pca",
            "compute_user_features",
            "merge_and_split",
        ]
        for t in expected_tasks:
            assert t in task_names, f"Missing task: {t}"
        assert len(task_names) == len(expected_tasks)

    def test_dag_has_trigger_operator(self):
        """Verify the DAG has a TriggerDagRunOperator for train_movielens."""
        dag_func = self.function_defs["etl_movielens"]
        has_trigger = any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Call)
                and isinstance(t.func, ast.Name)
                and t.func.id == "TriggerDagRunOperator"
                for t in ast.walk(node)
            )
            for node in dag_func.body
        )
        assert has_trigger, "Missing TriggerDagRunOperator in DAG body"

    def test_dag_has_markdown_doc(self):
        dag_func = self.function_defs["etl_movielens"]
        decorator = self._find_decorator(dag_func, "dag")
        assert decorator is not None, "Missing @dag decorator"
        keywords = {kw.arg for kw in decorator.keywords if kw.arg is not None}
        assert "doc_md" in keywords

    def test_dag_has_correct_dag_id(self):
        dag_func = self.function_defs["etl_movielens"]
        decorator = self._find_decorator(dag_func, "dag")
        assert decorator is not None
        for kw in decorator.keywords:
            if kw.arg == "dag_id":
                assert ast.literal_eval(kw.value) == "etl_movielens"

    def test_tasks_use_taskflow_decorator(self):
        dag_func = self.function_defs["etl_movielens"]
        for node in dag_func.body:
            if isinstance(node, ast.FunctionDef):
                decorator = self._find_decorator(node, "task")
                assert decorator is not None, f"Task {node.name} missing @task decorator"

    def test_dependency_chain_is_sequential(self):
        """Verify the dependency chain uses >> operator sequentially."""
        dag_func = self.function_defs["etl_movielens"]
        # Find the expression at the top level of the DAG function body
        # that contains the >> chain of tasks
        for node in ast.walk(dag_func):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.BinOp):
                if isinstance(node.value.op, ast.RShift):
                    parts = self._flatten_chain(node.value)
                    expected_chain = [
                        "download_data",
                        "sample_and_save_ratings",
                        "compute_movie_features",
                        "compute_genome_pca",
                        "compute_user_features",
                        "merge_and_split",
                        "trigger_train",
                    ]
                    assert parts == expected_chain, f"Chain mismatch: {parts}"
                    return
        pytest.fail("No >> dependency chain found in DAG body")

    def _find_decorator(self, func_node, name):
        for dec in func_node.decorator_list:
            if isinstance(dec, ast.Name) and dec.id == name:
                return dec
            if isinstance(dec, ast.Attribute) and dec.attr == name:
                return dec
            if isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name) and dec.func.id == name:
                    return dec
                if isinstance(dec.func, ast.Attribute) and dec.func.attr == name:
                    return dec
        return None

    def _flatten_chain(self, node):
        parts = []
        current = node
        while isinstance(current, ast.BinOp) and isinstance(current.op, ast.RShift):
            left = current.left
            right = current.right
            name = self._get_node_name(right)
            if name:
                parts.insert(0, name)
            current = left
        name = self._get_node_name(current)
        if name:
            parts.insert(0, name)
        return parts

    def _get_node_name(self, node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            return node.func.attr
        if isinstance(node, ast.Name):
            return node.id
        return None


def test_etl_dag_loads_without_errors():
    """Verify the DAG file is syntactically valid Python."""
    with open(DAG_PATH) as f:
        try:
            ast.parse(f.read())
        except SyntaxError as e:
            pytest.fail(f"Syntax error in DAG file: {e}")
