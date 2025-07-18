import sympy as sp
from latex2sympy2 import latex2sympy
import random
import json
import sys
import os
sys.set_int_max_str_digits(100000)

class NumericExpression:
    def __init__(self, expr_str, op_count=0, value=None, is_function=False):
        self.expr_str = expr_str
        self.op_count = op_count
        self.is_function=is_function
        self.value = value
        self.evalute()
    
    def evalute(self, re_calc=False):
        if self.value == None or re_calc:
            sp_expr = latex2sympy(self.expr_str)
            self.value = sp.N(sp_expr)
            
        return self.value

    def to_latex(self):
        return sp.latex(self.expr)

    def __str__(self):
        return f"NumericExpression('{self.expr}') [op_count={self.op_count}]"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        return self._binary_op(other, '+')

    def __sub__(self, other):
        return self._binary_op(other, '-')

    def __mul__(self, other):
        return self._binary_op(other, '*')

    def __truediv__(self, other):
        return self._binary_op(other, '/')

    def _binary_op(self, other, op):
        if isinstance(other, NumericExpression):
            left_str = self.expr_str
            right_str = other.expr_str
            if op == '*' or op == '/':
                if self.op_count >= 1 or self.is_function:
                    left_str = '(' + left_str + ')'
                if other.op_count >= 1:
                    right_str = '(' + right_str + ')'
            
            new_expr = f"{left_str}{op}{right_str}"
            new_count = self.op_count + other.op_count + 1
            return NumericExpression(new_expr, op_count=new_count, value=0)
        return None

def generate_zero_op_list():
    
    number_count = 100
    # 普通数字
    zero_op_list = [NumericExpression(str(i)) for i in range(number_count)]

    # 小数
    for denominator in range(2, 10):
        zero_op_list.extend([NumericExpression(f"\\frac{{{i}}}{{{denominator}}}") for i in range(1, number_count)])
    
    # 对数
    #### zero_op_list.extend([NumericExpression(f"\\log_{{2}}{{{i}}}", is_function=True) for i in range(2, number_count)])

    # 自然对数
    #### zero_op_list.extend([NumericExpression(f"\\ln{{{i}}}", is_function=True) for i in range(2, number_count)])

    # 三角函数
    #### angles = ["\\frac{\\pi}{2}", "\\frac{\\pi}{3}", "\\frac{\\pi}{4}", "\\frac{\\pi}{6}"]
    #### zero_op_list.extend([NumericExpression(f"\\sin{angle}") for angle in angles])
    #### zero_op_list.extend([NumericExpression(f"\\cos{angle}") for angle in angles])

    # 次方
    zero_op_list.extend([NumericExpression(f"{i}^2") for i in range(2, number_count)])
    zero_op_list.extend([NumericExpression(f"{i}^3") for i in range(2, number_count)])

    # 开方
    #### zero_op_list.extend([NumericExpression(f"\\sqrt[2]{{{i}}}") for i in range(2, number_count)])
    #### zero_op_list.extend([NumericExpression(f"\\sqrt[3]{{{i}}}") for i in range(2, number_count)])
    
    return zero_op_list


def combine_op_list(left_op_list, right_op_list, target_count=1200):
    one_op_list = []
    ops = ['+', '-', '*', '/']

    while len(one_op_list) < target_count:
        # 随机选择操作数和操作符
        a = random.choice(left_op_list)
        b = random.choice(right_op_list)
        op = random.choice(ops)
        # 随机交换a和b
        if random.random() < 0.5:
            a, b = b, a
        
        try:
            if op == '+':
                new_expr = a + b
            elif op == '-':
                new_expr = a - b
            elif op == '*':
                new_expr = a * b
            elif op == '/':
                new_expr = a / b

            if new_expr is not None:
                one_op_list.append(new_expr)
        except Exception as e:
            print(f"遇到异常，继续进行{e}")
    
    return one_op_list

def save_list(op_list, file_name, max_count=1000):
    random.shuffle(op_list)

    # 这里可以替换为其他前缀
    prompt = "Evaluate this LaTeX numerical expression step-by-step and give the final value within \\boxed{}: "
    qa_data = []
    for i, expr in enumerate(op_list):
        if len(qa_data) >= max_count:
            break
        
        expr_value = expr.evalute(re_calc=True)
        if expr_value.is_real and expr_value.is_finite:
            # 确保保存的答案与真实值不要太大误差
            answer_str = str(expr_value)
            if abs(float(answer_str) - float(expr_value)) <0.001:
                qa_data.append({
                    "id": i,
                    "prompt": prompt + expr.expr_str,
                    "answer": answer_str
                })
        else:
            print(f"不能转换为float（可能是无穷或未定义）：{expr_value}")
            
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=4, ensure_ascii=False)

    print(f"文件已保存：{file_name}，数量：{len(qa_data)}")


if __name__ == "__main__":
    base_dir = "result/"
    # 判断目录是否存在，不存在则创建
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    zero_op_list = generate_zero_op_list()
    save_list(zero_op_list, base_dir + "00_op_list.json")
    cache_list = [zero_op_list]

    n = 20
    for i in range(1, n + 1):
        result_list = []
        print(f"生成{i}计算步骤表达式")
        for j in range(i // 2 + i % 2):
            print(f"\t组合 {j}+1+{i - 1 - j}")
            result_list.extend(combine_op_list(cache_list[j], cache_list[i - 1 - j]))
        cache_list.append(result_list)
        save_list(result_list, base_dir + f"{i:02d}_op_list.json")

    print("运行结束")
