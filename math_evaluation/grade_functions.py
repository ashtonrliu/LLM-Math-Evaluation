from sympy import simplify
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy
import sympy as sp

def grade_custom_math_problem(pred_latex, correct_answer):
    if pred_latex is None:
        return 0.0
    
    corr_latex = correct_answer.strip()
    print(f"check_math_answer: {pred_latex} <---> {correct_answer}")

    if pred_latex.replace(" ", "") == corr_latex.replace(" ", ""):
        print("check_math_answer: exact match")
        return 0.0

    try:
        # 一般表达式
        pred_expr = latex2sympy(pred_latex)
        corr_expr = latex2sympy(corr_latex)
        pred_value = sp.N(pred_expr)
        corr_value = sp.N(corr_expr)
        diff = abs(simplify(pred_value - corr_value))
        print(f"\t\tcheck_math_answer: pred_value: {pred_value}  corr_value:{corr_value}  diff={diff}")

        if diff > 1.0:
            reward = 0.0
        else:
            reward = 1 - diff
        return float(reward)

    except Exception as e:
        print(f"check_math_answer: SymPy Exception: {type(e).__name__}: {e}")
        return 0.0
