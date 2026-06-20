"""Structural tests for train_movielens DAG using AST analysis."""

import ast
from pathlib import Path

import pytest

DAG_PATH = Path(__file__).resolve().parent.parent / "airflow" / "dags" / "train_movielens.py"


class TestTrainDagStructure:
    """Verify train_movielens.py defines the expected pipeline structure."""

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
        assert "train_movielens" in self.function_defs

    def test_dag_has_one_task(self):
        dag_func = self.function_defs["train_movielens"]
        task_names = [
            node.name
            for node in dag_func.body
            if isinstance(node, ast.FunctionDef)
        ]
        assert task_names == ["train_and_register_model"]

    def test_dag_has_trigger_operator(self):
        dag_func = self.function_defs["train_movielens"]
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
        assert has_trigger

    def test_dag_has_correct_dag_id(self):
        dag_func = self.function_defs["train_movielens"]
        decorator = self._find_decorator(dag_func, "dag")
        assert decorator is not None
        for kw in decorator.keywords:
            if kw.arg == "dag_id":
                assert ast.literal_eval(kw.value) == "train_movielens"

    def test_task_uses_virtualenv_decorator(self):
        dag_func = self.function_defs["train_movielens"]
        for node in dag_func.body:
            if isinstance(node, ast.FunctionDef):
                decorator = self._find_decorator(node, "virtualenv")
                assert decorator is not None

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


def test_train_dag_loads_without_errors():
    """Verify the DAG file is syntactically valid Python."""
    with open(DAG_PATH) as f:
        try:
            ast.parse(f.read())
        except SyntaxError as e:
            pytest.fail(f"Syntax error in DAG file: {e}")
