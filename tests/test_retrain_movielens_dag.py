"""Structural tests for retrain_movielens DAG using AST analysis."""

import ast
from pathlib import Path

import pytest

DAG_PATH = Path(__file__).resolve().parent.parent / "airflow" / "dags" / "retrain_movielens.py"


class TestRetrainDagStructure:
    """Verify retrain_movielens.py defines the expected pipeline structure."""

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
        assert "retrain_movielens" in self.function_defs

    def test_dag_has_two_tasks(self):
        dag_func = self.function_defs["retrain_movielens"]
        task_names = [
            node.name
            for node in dag_func.body
            if isinstance(node, ast.FunctionDef)
        ]
        expected = ["train_challenger", "evaluate_and_promote"]
        for t in expected:
            assert t in task_names
        assert len(task_names) == len(expected)

    def test_dag_has_correct_dag_id(self):
        dag_func = self.function_defs["retrain_movielens"]
        decorator = self._find_decorator(dag_func, "dag")
        assert decorator is not None
        for kw in decorator.keywords:
            if kw.arg == "dag_id":
                assert ast.literal_eval(kw.value) == "retrain_movielens"

    def test_train_challenger_uses_virtualenv(self):
        dag_func = self.function_defs["retrain_movielens"]
        for node in dag_func.body:
            if isinstance(node, ast.FunctionDef) and node.name == "train_challenger":
                decorator = self._find_decorator(node, "virtualenv")
                assert decorator is not None

    def test_evaluate_and_promote_uses_task(self):
        dag_func = self.function_defs["retrain_movielens"]
        for node in dag_func.body:
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate_and_promote":
                decorator = self._find_decorator(node, "task")
                assert decorator is not None, "evaluate_and_promote should use @task (not virtualenv)"

    def _find_decorator(self, func_node, name):
        for dec in func_node.decorator_list:
            dec_name = None
            if isinstance(dec, ast.Name):
                dec_name = dec.id
            elif isinstance(dec, ast.Attribute):
                dec_name = dec.attr
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    dec_name = dec.func.id
                elif isinstance(dec.func, ast.Attribute):
                    dec_name = dec.func.attr
            if dec_name == name:
                return dec
        return None


def test_retrain_dag_loads_without_errors():
    """Verify the DAG file is syntactically valid Python."""
    with open(DAG_PATH) as f:
        try:
            ast.parse(f.read())
        except SyntaxError as e:
            pytest.fail(f"Syntax error in DAG file: {e}")
