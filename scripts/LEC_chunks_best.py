import time
import sys
import argparse
import random
import copy
import itertools
import functools
import subprocess
import math
from itertools import combinations, product
from statistics import mean, median, variance

from mpi4py import MPI

print = functools.partial(print, flush=True)


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", nargs="?")
    parser.add_argument(
        "-ld",
        "--lenghdecompose",
        nargs="?",
        type=str,
        default=2,
        help="Lenght of chuncks of input vars",
    )
    parser.add_argument("-lim", "--limit", nargs="?", type=int, default=10000)
    parser.add_argument("-si", "--shuffle_inputs", nargs="?", type=int, default=False)
    parser.add_argument(
        "-s",
        "--solver",
        nargs="?",
        type=str,
        default="kissat2022",
        help="SAT Solver",
    )
    return parser


class Error(Exception):
    pass


def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type("Enum", (), enums)


def round_up(number):
    return int(number) + (number % 1 > 0)


def getCNF(from_file):
    dimacs = open(from_file).readlines()
    clauses = []
    header = None
    comments = []
    inputs = []
    vars_left = []
    outputs_first = []
    vars_right = []
    outputs_second = []
    miter_vars = []
    cubes_vars = []
    outputs = []
    gates_vars = []
    var_set = []
    for i in range(len(dimacs)):
        if dimacs[i][0] == "p":
            header = dimacs[i][:-1] if dimacs[i][-1] == "\n" else dimacs[i]
        elif dimacs[i][0] == "c":
            comments.append(dimacs[i][:-1] if dimacs[i][-1] == "\n" else dimacs[i])
            if "c inputs: " in dimacs[i]:
                inputs = list(map(int, dimacs[i].split(":")[1].split()))
            elif "c variables for gates in first scheme" in dimacs[i]:
                vars_right = [
                    x
                    for x in range(
                        int(dimacs[i].split()[-3]), int(dimacs[i].split()[-1]) + 1
                    )
                ]
            elif "c outputs first scheme" in dimacs[i]:
                outputs_first = list(map(int, dimacs[i].split(":")[1].split()))
            elif "c variables for gates in second scheme" in dimacs[i]:
                vars_left = [
                    x
                    for x in range(
                        int(dimacs[i].split()[-3]), int(dimacs[i].split()[-1]) + 1
                    )
                ]
            elif "c outputs second scheme" in dimacs[i]:
                outputs_second = list(map(int, dimacs[i].split(":")[1].split()))
            elif "c miter variables" in dimacs[i]:
                miter_vars = list(map(int, dimacs[i].split(":")[1].split()))
            elif "c cubes variables:" in dimacs[i]:
                cubes_vars = list(map(int, dimacs[i].split(":")[1].split()))
            elif "c outputs: " in dimacs[i]:
                outputs = list(map(int, dimacs[i].split(":")[1].split()))
            elif "c variables for gates:" in dimacs[i]:
                gates_vars = [
                    x
                    for x in range(
                        int(dimacs[i].split()[-3]), int(dimacs[i].split()[-1]) + 1
                    )
                ]
            elif "c var_set:" in dimacs[i]:
                var_set = list(map(int, dimacs[i].split(":")[1].split()))
        else:
            if len(dimacs[i]) > 1:
                clauses.append(list(map(int, dimacs[i].split()[:-1])))
    return (
        header,
        comments,
        inputs,
        outputs,
        gates_vars,
        vars_left,
        outputs_first,
        vars_right,
        outputs_second,
        miter_vars,
        cubes_vars,
        var_set,
        clauses,
    )


def make_pairs(*lists):
    for t in combinations(lists, 2):
        for pair in product(*t):
            yield pair


def create_and_solve_CNF(task_index, clauses, new_clauses, max_var, solver):
    cnf_str = create_str_CNF(clauses, new_clauses, max_var)
    # with open('lec_CvK_16_ld4_rand_subtask' + str(task_index) + '.cnf', 'w') as f:
    # print(cnf_str, file = f)
    result = solve_CNF(cnf_str, solver)
    answer = "INDET"
    time = None
    conflicts = None
    for line in result:
        if len(line) > 0 and line[0] == "s":
            if "UNSAT" in line:
                answer = "UNSAT"
            elif "SAT" in line:
                answer = "SAT"
        elif (
            ("c process-time" in line and "kissat" in solver)
            or ("c total process time" in line and "cadical" in solver)
            or ("c CPU time" in line)
        ):
            time = float(line.split()[-2])
        elif ("c conflicts:" in line and "kissat" in solver) or (
            "c conflicts:" in line and "cadical" in solver
        ):
            conflicts = int(line.split()[-4])
        elif "c conflicts " in line:
            conflicts = int(line.split()[3])
    # if task_index < 180 and answer == 'SAT':
    #  with open('a5_1_64_11_d3_t' + str(task_index) + '.cnf', 'w') as f:
    #    print(cnf_str, file = f)
    return answer, time, conflicts


def create_str_CNF(clauses, new_clauses, max_var):
    lines = []
    header_ = "p cnf " + str(max_var) + " " + str(len(clauses) + len(new_clauses))
    lines.append(header_)
    lines.extend([" ".join(list(map(str, clause))) + " 0" for clause in clauses])
    lines.extend([" ".join(list(map(str, clause))) + " 0" for clause in new_clauses])
    cnf = "\n".join(lines)
    # pprint.pprint(cnf)
    return cnf


def solve_CNF(cnf, solver):
    config = [solver]
    best_params = "--ands=1 --backbone=2 --bump=1 --bumpreasons=1 --chrono=1 --compact=0 --decay=10 --definitions=0 --eliminate=1 --eliminatebound=10 --equivalences=0 --extract=1 --forcephase=1 --forward=0 --ifthenelse=1 --minimize=0 --minimizedepth=16 --minimizeticks=1 --otfs=0 --phase=1 --phasesaving=1 --probe=1 --promote=0 --reduce=0 --reducefraction=50 --reduceinit=256 --reduceint=10 --reluctant=0 --rephase=0 --restartint=1000 --restartmargin=10 --shrink=0 --simplify=1 --stable=2 --substitute=1 --sweep=1 --time=20 --tumble=0 --vivify=0 --walkinitially=1 --warmup=0"
    config += best_params.split()
    solver = subprocess.run(config, capture_output=True, text=True, input=cnf)
    result = solver.stdout.split("\n")
    errors = solver.stderr
    # print(result)
    if len(errors) > 0:
        print(errors)
    return result


def solve_CNF_timelimit(cnf, solver, timelim):
    solver = subprocess.run(
        [solver, f"--time={timelim}"],
        capture_output=True,
        text=True,
        input=cnf,
    )
    result = solver.stdout.split("\n")
    errors = solver.stderr
    # print(result)
    if len(errors) > 0:
        print(errors)
    return result


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield tuple(lst[i : i + n])


def solve_vector(
    task_index,
    binaryvector,
    input_decompose,
    clauses,
    max_var,
    solver,
    force_xor_flag,
    force_disj_flag,
):
    if len(binaryvector) == len(input_decompose):
        new_clauses = []
        current_var = max_var
        for bit, inp_chunk in zip(binaryvector, input_decompose):
            if bit == 1:
                if len(inp_chunk) == 2:
                    if force_disj_flag == False:
                        new_clauses_, current_var = encode_XOR_clauses(
                            inp_chunk, current_var
                        )
                    else:
                        new_clauses_, current_var = encode_DISJUNCTION_clauses(
                            inp_chunk, current_var
                        )
                    new_clauses.extend(new_clauses_)
                    new_clauses.append([current_var])
                elif len(inp_chunk) == 3:
                    if force_xor_flag == False:
                        new_clauses_, current_var = encode_MAJORITY_clauses(
                            inp_chunk, current_var
                        )
                    else:
                        new_clauses_, current_var = encode_TRIPLEXOR_clauses(
                            inp_chunk, current_var
                        )
                    new_clauses.extend(new_clauses_)
                    new_clauses.append([current_var])
                elif len(inp_chunk) == 4:
                    if force_xor_flag == False:
                        new_clauses_, current_var = encode_BENT_clauses(
                            inp_chunk, current_var
                        )
                    else:
                        new_clauses_, current_var = encode_QUADROXOR_clauses(
                            inp_chunk, current_var
                        )
                    new_clauses.extend(new_clauses_)
                    new_clauses.append([current_var])
                elif len(inp_chunk) == 5:
                    new_clauses_, current_var = encode_PENTAXOR_clauses(
                        inp_chunk, current_var
                    )
                    new_clauses.extend(new_clauses_)
                    new_clauses.append([current_var])
                elif len(inp_chunk) == 8:
                    (
                        new_clauses_,
                        current_var,
                    ) = encode_clauses_by_vector_function_for_8_vars(
                        inp_chunk, current_var
                    )
                    new_clauses.extend(new_clauses_)
                    new_clauses.append([current_var])
            elif bit == 0:
                if len(inp_chunk) == 2:
                    if force_disj_flag == False:
                        new_clauses_, current_var = encode_XOR_clauses(
                            inp_chunk, current_var
                        )
                    else:
                        new_clauses_, current_var = encode_DISJUNCTION_clauses(
                            inp_chunk, current_var
                        )
                    new_clauses.extend(new_clauses_)
                    new_clauses.append([-current_var])
                elif len(inp_chunk) == 3:
                    if force_xor_flag == False:
                        new_clauses_, current_var = encode_MAJORITY_clauses(
                            inp_chunk, current_var
                        )
                    else:
                        new_clauses_, current_var = encode_TRIPLEXOR_clauses(
                            inp_chunk, current_var
                        )
                    new_clauses.extend(new_clauses_)
                    new_clauses.append([-current_var])
                elif len(inp_chunk) == 4:
                    if force_xor_flag == False:
                        new_clauses_, current_var = encode_BENT_clauses(
                            inp_chunk, current_var
                        )
                    else:
                        new_clauses_, current_var = encode_QUADROXOR_clauses(
                            inp_chunk, current_var
                        )
                    new_clauses.extend(new_clauses_)
                    new_clauses.append([-current_var])
                elif len(inp_chunk) == 5:
                    new_clauses_, current_var = encode_PENTAXOR_clauses(
                        inp_chunk, current_var
                    )
                    new_clauses.extend(new_clauses_)
                    new_clauses.append([-current_var])
                elif len(inp_chunk) == 8:
                    (
                        new_clauses_,
                        current_var,
                    ) = encode_clauses_by_vector_function_for_8_vars(
                        inp_chunk, current_var
                    )
                    new_clauses.extend(new_clauses_)
                    new_clauses.append([-current_var])
        answer, time, conflicts = create_and_solve_CNF(
            task_index, clauses, new_clauses, current_var, solver
        )
        return answer, time, conflicts
    else:
        raise "len(binaryvector) != len(input_decompose)"


def encode_DISJUNCTION_clauses(inp_pair, current_var):
    current_var += 1
    # fmt: off
    clauses = [
        [ -current_var,  inp_pair[0],  inp_pair[1] ],
        [  current_var,  inp_pair[0], -inp_pair[1] ],
        [ -current_var,  inp_pair[0],  inp_pair[1] ],
        [  current_var, -inp_pair[0], -inp_pair[1] ],
    ]
    # fmt: on
    return clauses, current_var


def encode_XOR_clauses(inp_pair, current_var):
    current_var += 1
    # fmt: off
    clauses = [
        [  current_var, -inp_pair[0],  inp_pair[1] ],
        [  current_var,  inp_pair[0], -inp_pair[1] ],
        [ -current_var,  inp_pair[0],  inp_pair[1] ],
        [ -current_var, -inp_pair[0], -inp_pair[1] ],
    ]
    # fmt: on
    return clauses, current_var


def encode_MAJORITY_clauses(inp_triple, current_var):
    current_var += 1
    # fmt: off
    clauses = [
        [ -current_var,  inp_triple[0],  inp_triple[1],  inp_triple[2] ],
        [ -current_var,  inp_triple[0],  inp_triple[1], -inp_triple[2] ],
        [ -current_var,  inp_triple[0], -inp_triple[1],  inp_triple[2] ],
        [  current_var,  inp_triple[0], -inp_triple[1], -inp_triple[2] ],
        [ -current_var, -inp_triple[0],  inp_triple[1],  inp_triple[2] ],
        [  current_var, -inp_triple[0],  inp_triple[1], -inp_triple[2] ],
        [  current_var, -inp_triple[0], -inp_triple[1],  inp_triple[2] ],
        [  current_var, -inp_triple[0], -inp_triple[1], -inp_triple[2] ],
    ]
    # fmt: on
    return clauses, current_var


def encode_BENT_clauses(inp_cuadro, current_var):
    current_var += 1
    # fmt: off
    clauses = [
        [ -current_var,  inp_cuadro[0],  inp_cuadro[1],  inp_cuadro[2],  inp_cuadro[3] ],
        [ -current_var,  inp_cuadro[0],  inp_cuadro[1],  inp_cuadro[2], -inp_cuadro[3] ],
        [ -current_var,  inp_cuadro[0],  inp_cuadro[1], -inp_cuadro[2],  inp_cuadro[3] ],
        [ -current_var,  inp_cuadro[0],  inp_cuadro[1], -inp_cuadro[2], -inp_cuadro[3] ],
        [ -current_var,  inp_cuadro[0], -inp_cuadro[1],  inp_cuadro[2],  inp_cuadro[3] ],
        [  current_var,  inp_cuadro[0], -inp_cuadro[1],  inp_cuadro[2], -inp_cuadro[3] ],
        [ -current_var,  inp_cuadro[0], -inp_cuadro[1], -inp_cuadro[2],  inp_cuadro[3] ],
        [  current_var,  inp_cuadro[0], -inp_cuadro[1], -inp_cuadro[2], -inp_cuadro[3] ],
        [ -current_var, -inp_cuadro[0],  inp_cuadro[1],  inp_cuadro[2],  inp_cuadro[3] ],
        [ -current_var, -inp_cuadro[0],  inp_cuadro[1],  inp_cuadro[2], -inp_cuadro[3] ],
        [  current_var, -inp_cuadro[0],  inp_cuadro[1], -inp_cuadro[2],  inp_cuadro[3] ],
        [  current_var, -inp_cuadro[0],  inp_cuadro[1], -inp_cuadro[2], -inp_cuadro[3] ],
        [ -current_var, -inp_cuadro[0], -inp_cuadro[1],  inp_cuadro[2],  inp_cuadro[3] ],
        [  current_var, -inp_cuadro[0], -inp_cuadro[1],  inp_cuadro[2], -inp_cuadro[3] ],
        [  current_var, -inp_cuadro[0], -inp_cuadro[1], -inp_cuadro[2],  inp_cuadro[3] ],
        [ -current_var, -inp_cuadro[0], -inp_cuadro[1], -inp_cuadro[2], -inp_cuadro[3] ],
    # fmt: on
    ]
    return clauses, current_var


def encode_TRIPLEXOR_clauses(inp_chunk, current_var):
    current_var += 1
    # fmt: off
    clauses = [
        [ -current_var,  inp_chunk[0],  inp_chunk[1],  inp_chunk[2] ],
        [  current_var,  inp_chunk[0],  inp_chunk[1], -inp_chunk[2] ],
        [  current_var,  inp_chunk[0], -inp_chunk[1],  inp_chunk[2] ],
        [ -current_var,  inp_chunk[0], -inp_chunk[1], -inp_chunk[2] ],
        [  current_var, -inp_chunk[0],  inp_chunk[1],  inp_chunk[2] ],
        [ -current_var, -inp_chunk[0],  inp_chunk[1], -inp_chunk[2] ],
        [ -current_var, -inp_chunk[0], -inp_chunk[1],  inp_chunk[2] ],
        [  current_var, -inp_chunk[0], -inp_chunk[1], -inp_chunk[2] ],
    ]
    # fmt: on
    return clauses, current_var


def encode_QUADROXOR_clauses(inp_cuadro, current_var):
    current_var += 1
    # fmt: off
    clauses = [
        [ -current_var,  inp_cuadro[0],  inp_cuadro[1],  inp_cuadro[2],  inp_cuadro[3] ],
        [  current_var,  inp_cuadro[0],  inp_cuadro[1],  inp_cuadro[2], -inp_cuadro[3] ],
        [  current_var,  inp_cuadro[0],  inp_cuadro[1], -inp_cuadro[2],  inp_cuadro[3] ],
        [ -current_var,  inp_cuadro[0],  inp_cuadro[1], -inp_cuadro[2], -inp_cuadro[3] ],
        [  current_var,  inp_cuadro[0], -inp_cuadro[1],  inp_cuadro[2],  inp_cuadro[3] ],
        [ -current_var,  inp_cuadro[0], -inp_cuadro[1],  inp_cuadro[2], -inp_cuadro[3] ],
        [ -current_var,  inp_cuadro[0], -inp_cuadro[1], -inp_cuadro[2],  inp_cuadro[3] ],
        [  current_var,  inp_cuadro[0], -inp_cuadro[1], -inp_cuadro[2], -inp_cuadro[3] ],
        [  current_var, -inp_cuadro[0],  inp_cuadro[1],  inp_cuadro[2],  inp_cuadro[3] ],
        [ -current_var, -inp_cuadro[0],  inp_cuadro[1],  inp_cuadro[2], -inp_cuadro[3] ],
        [ -current_var, -inp_cuadro[0],  inp_cuadro[1], -inp_cuadro[2],  inp_cuadro[3] ],
        [  current_var, -inp_cuadro[0],  inp_cuadro[1], -inp_cuadro[2], -inp_cuadro[3] ],
        [ -current_var, -inp_cuadro[0], -inp_cuadro[1],  inp_cuadro[2],  inp_cuadro[3] ],
        [  current_var, -inp_cuadro[0], -inp_cuadro[1],  inp_cuadro[2], -inp_cuadro[3] ],
        [  current_var, -inp_cuadro[0], -inp_cuadro[1], -inp_cuadro[2],  inp_cuadro[3] ],
        [ -current_var, -inp_cuadro[0], -inp_cuadro[1], -inp_cuadro[2], -inp_cuadro[3] ],
    ]
    # fmt: on
    return clauses, current_var


def encode_PENTAXOR_clauses(inp_penta, current_var):
    current_var += 1
    # fmt: off
    clauses = [
        [ -current_var,  inp_penta[0],  inp_penta[1],  inp_penta[2],  inp_penta[3],  inp_penta[4] ],
        [  current_var,  inp_penta[0],  inp_penta[1],  inp_penta[2],  inp_penta[3], -inp_penta[4] ],
        [  current_var,  inp_penta[0],  inp_penta[1],  inp_penta[2], -inp_penta[3],  inp_penta[4] ],
        [ -current_var,  inp_penta[0],  inp_penta[1],  inp_penta[2], -inp_penta[3], -inp_penta[4] ],
        [  current_var,  inp_penta[0],  inp_penta[1], -inp_penta[2],  inp_penta[3],  inp_penta[4] ],
        [ -current_var,  inp_penta[0],  inp_penta[1], -inp_penta[2],  inp_penta[3], -inp_penta[4] ],
        [ -current_var,  inp_penta[0],  inp_penta[1], -inp_penta[2], -inp_penta[3],  inp_penta[4] ],
        [  current_var,  inp_penta[0],  inp_penta[1], -inp_penta[2], -inp_penta[3], -inp_penta[4] ],
        [  current_var,  inp_penta[0], -inp_penta[1],  inp_penta[2],  inp_penta[3],  inp_penta[4] ],
        [ -current_var,  inp_penta[0], -inp_penta[1],  inp_penta[2],  inp_penta[3], -inp_penta[4] ],
        [ -current_var,  inp_penta[0], -inp_penta[1],  inp_penta[2], -inp_penta[3],  inp_penta[4] ],
        [  current_var,  inp_penta[0], -inp_penta[1],  inp_penta[2], -inp_penta[3], -inp_penta[4] ],
        [ -current_var,  inp_penta[0], -inp_penta[1], -inp_penta[2],  inp_penta[3],  inp_penta[4] ],
        [  current_var,  inp_penta[0], -inp_penta[1], -inp_penta[2],  inp_penta[3], -inp_penta[4] ],
        [  current_var,  inp_penta[0], -inp_penta[1], -inp_penta[2], -inp_penta[3],  inp_penta[4] ],
        [ -current_var,  inp_penta[0], -inp_penta[1], -inp_penta[2], -inp_penta[3], -inp_penta[4] ],
        [  current_var, -inp_penta[0],  inp_penta[1],  inp_penta[2],  inp_penta[3],  inp_penta[4] ],
        [ -current_var, -inp_penta[0],  inp_penta[1],  inp_penta[2],  inp_penta[3], -inp_penta[4] ],
        [ -current_var, -inp_penta[0],  inp_penta[1],  inp_penta[2], -inp_penta[3],  inp_penta[4] ],
        [  current_var, -inp_penta[0],  inp_penta[1],  inp_penta[2], -inp_penta[3], -inp_penta[4] ],
        [ -current_var, -inp_penta[0],  inp_penta[1], -inp_penta[2],  inp_penta[3],  inp_penta[4] ],
        [  current_var, -inp_penta[0],  inp_penta[1], -inp_penta[2],  inp_penta[3], -inp_penta[4] ],
        [  current_var, -inp_penta[0],  inp_penta[1], -inp_penta[2], -inp_penta[3],  inp_penta[4] ],
        [ -current_var, -inp_penta[0],  inp_penta[1], -inp_penta[2], -inp_penta[3], -inp_penta[4] ],
        [ -current_var, -inp_penta[0], -inp_penta[1],  inp_penta[2],  inp_penta[3],  inp_penta[4] ],
        [  current_var, -inp_penta[0], -inp_penta[1],  inp_penta[2],  inp_penta[3], -inp_penta[4] ],
        [  current_var, -inp_penta[0], -inp_penta[1],  inp_penta[2], -inp_penta[3],  inp_penta[4] ],
        [ -current_var, -inp_penta[0], -inp_penta[1],  inp_penta[2], -inp_penta[3], -inp_penta[4] ],
        [  current_var, -inp_penta[0], -inp_penta[1], -inp_penta[2],  inp_penta[3],  inp_penta[4] ],
        [ -current_var, -inp_penta[0], -inp_penta[1], -inp_penta[2],  inp_penta[3], -inp_penta[4] ],
        [ -current_var, -inp_penta[0], -inp_penta[1], -inp_penta[2], -inp_penta[3],  inp_penta[4] ],
        [  current_var, -inp_penta[0], -inp_penta[1], -inp_penta[2], -inp_penta[3], -inp_penta[4] ],
    ]
    # fmt: on
    return clauses, current_var


def encode_clauses_by_vector_function_for_8_vars(inp_chunk, current_var):
    current_var += 1
    # fmt: off
    vector_func = [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    # fmt: on
    truth_table_lines = list(itertools.product([0, 1], repeat=len(inp_chunk) + 1))
    truth_table_lines_with_vec = zip(truth_table_lines, vector_func)
    truth_table_zero_lines = [(x) for x, y in truth_table_lines_with_vec if y == 0]
    clauses = []
    for line in truth_table_zero_lines:
        if len(line) != len(inp_chunk) + 1:
            raise Error("len(line) != len(inp_chunk) + 1")
        current_var_ = current_var if line[-1] == 0 else -current_var
        clause = [current_var_]
        for i in range(len(inp_chunk)):
            inp_ = inp_chunk[i] if line[i] == 0 else -inp_chunk[i]
            clause.append(inp_)
        clauses.append(clause)
    return clauses, current_var


def random_binary_vector(n):
    return tuple([random.randint(0, 1) for i in range(n)])


def make_template_clauses(clauses, miter_vars, outputs):
    template_clauses = []
    removed_clauses = []
    for clause in clauses:
        if (len(miter_vars) > 0) and (
            len([lit_ for lit_ in clause if abs(lit_) in miter_vars]) < len(miter_vars)
        ):
            template_clauses.append(clause)
        elif (
            (len(miter_vars) == 0)
            and (len(clause) > 1)
            and (len([lit_ for lit_ in clause if abs(lit_) in outputs]) != len(clause))
        ):
            template_clauses.append(clause)
        else:
            removed_clauses.append(clause)
    return template_clauses, removed_clauses


######################################################################################################
##-----------------------------------------------MAIN-----------------------------------------------##
######################################################################################################


if __name__ == "__main__":
    # Define MPI message tags
    tags = enum("READY", "START", "DONE", "EXIT")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.size
    status = MPI.Status()
    if rank == 0:
        start_time = MPI.Wtime()
        parser = createParser()
        namespace = parser.parse_args(sys.argv[1:])
        # cnf = [0 header, 1 inputs, 2 vars_first, 3 outputs_first, 4 vars_second, 5 outputs_second, 6 mut_gate, 7 mut_var, 8 clauses]
        filename = namespace.name
        shuffle_inputs = False if namespace.shuffle_inputs == False else True
        cnf_name = "".join(filename.split("/")[-1].split(".")[:-1])
        nof_additional_chunks = 0
        force_xor_flag = False
        force_disj_flag = False
        if "_" in namespace.lenghdecompose:
            len_inp_chunks = [int(namespace.lenghdecompose.split("_")[0])]
            nof_additional_chunks = int(namespace.lenghdecompose.split("_")[1])
        elif "+" in namespace.lenghdecompose:
            len_inp_chunks = list(map(int, namespace.lenghdecompose.split("+")))
        elif "x" in namespace.lenghdecompose:
            len_inp_chunks = [int(namespace.lenghdecompose.split("x")[0])]
            force_xor_flag = True
        elif "d" in namespace.lenghdecompose:
            len_inp_chunks = [int(namespace.lenghdecompose.split("d")[0])]
            force_disj_flag = True
        else:
            len_inp_chunks = [int(namespace.lenghdecompose)]
        tasklimit = namespace.limit
        solver = namespace.solver

        (
            header,
            comments,
            inputs,
            outputs,
            gates_vars,
            vars_left,
            outputs_first,
            vars_right,
            outputs_second,
            miter_vars,
            current_buckets,
            var_set,
            clauses,
        ) = getCNF(from_file=filename)
        max_var = int(header.split()[2])
        template_clauses, removed_clauses = make_template_clauses(
            clauses, miter_vars, outputs
        )

        # Shuffle vars for chunks
        vars_for_chunks = copy.copy(inputs)
        if shuffle_inputs == True:
            random.shuffle(vars_for_chunks)

        # Create chunks
        input_decompose = []
        for i_ in len_inp_chunks:
            input_decompose_ = list(chunks(vars_for_chunks, i_))
            input_decompose_[-1] = vars_for_chunks[-i_:]
            input_decompose += input_decompose_

        # Additional chunks, if necessary
        i = 0
        while i < nof_additional_chunks:
            new_chunk = tuple(sorted(random.sample(vars_for_chunks, len_inp_chunks[0])))
            if new_chunk not in input_decompose:
                input_decompose.append(new_chunk)
                i += 1

        # Create binary vectors
        full_solve_flag = False
        total_tasks = pow(2, len(input_decompose))
        if tasklimit > total_tasks:
            tasklimit = total_tasks
            full_solve_flag = True
            binary_vectors = list(
                itertools.product([0, 1], repeat=len(input_decompose))
            )

    else:
        solver = None
        input_decompose = None
        max_var = None
        clauses = None
        force_xor_flag = None
        force_disj_flag = None
        # template_clauses= None

    force_xor_flag = comm.bcast(force_xor_flag, root=0)
    force_disj_flag = comm.bcast(force_disj_flag, root=0)
    solver = comm.bcast(solver, root=0)
    input_decompose = comm.bcast(input_decompose, root=0)
    clauses = comm.bcast(clauses, root=0)
    # template_clauses = comm.bcast(template_clauses, root=0)
    max_var = comm.bcast(max_var, root=0)

    if rank == 0:
        # Master process executes code below
        task_index = 0
        num_workers = size - 1
        sats = []
        unsats = []
        closed_workers = 0
        results_table = []
        filename = (
            "tmp_log_ID_"
            + str(namespace.lenghdecompose)
            + "_"
            + cnf_name
            + "_"
            + str(tasklimit)
            + "_"
            + solver
            + ".log"
        )
        with open(filename, "w") as f:
            print(
                "Master starting with %d workers and %d tasks"
                % (num_workers, tasklimit),
                file=f,
            )
            while closed_workers < num_workers:
                data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                source = status.Get_source()
                tag = status.Get_tag()
                if tag == tags.READY:
                    # Worker is ready, so send it a task
                    if task_index < tasklimit:
                        print("", file=f)
                        if full_solve_flag == True:
                            task = (task_index, binary_vectors[task_index])
                        else:
                            task = (
                                task_index,
                                random_binary_vector(len(input_decompose)),
                            )
                        comm.send(task, dest=source, tag=tags.START)
                        print(
                            "Sending task %d to worker %d (current runtime % 6.2f)"
                            % (task_index, source, (MPI.Wtime() - start_time)),
                            file=f,
                        )
                        print(task, file=f)
                        task_index += 1
                    else:
                        comm.send(None, dest=source, tag=tags.EXIT)
                elif tag == tags.DONE:
                    print("", file=f)
                    print(
                        "Got data from worker %d (current runtime % 6.2f)"
                        % (source, (MPI.Wtime() - start_time)),
                        file=f,
                    )
                    print(data, file=f)
                    results_table.append(data)
                    if "UNSAT" in data[3]:
                        unsats.append(data[1])
                    elif "SAT" in data[3]:
                        sats.append(data[1])
                    current_avg_solvetime = round(
                        sum([x[1] for x in results_table]) / len(results_table), 2
                    )
                    current_avg_conflicts = round(
                        sum([x[2] for x in results_table]) / len(results_table), 2
                    )
                    print(
                        "Current time estimate: ",
                        current_avg_solvetime * total_tasks,
                        "Current ratio:",
                        (current_avg_conflicts * total_tasks) / pow(2, len(inputs)),
                        file=f,
                    )
                elif tag == tags.EXIT:
                    print("Worker %d exited." % source, file=f)
                    closed_workers += 1
            print("Master finishing", file=f)
            print(
                "Total runtime:", MPI.Wtime() - start_time, "on", size, "cores", file=f
            )
            print("", file=f)
            res_time_ = [x[1] for x in results_table]
            avg_solvetime = round(sum(res_time_) / len(results_table), 2)
            res_conflicts_ = [x[2] for x in results_table]
            avg_conflicts = round(sum(res_conflicts_) / len(results_table), 2)
            if full_solve_flag == True:
                print("All possible tasks solved.", file=f)
            print("CNF name:", cnf_name, file=f)
            print("Decomposition type:", namespace.lenghdecompose, file=f)
            print("Solver:", solver, file=f)
            print("Solved (all)", len(results_table), "tasks.", file=f)
            print("Total tasks:", total_tasks, file=f)
            print("Average solvetime:", avg_solvetime, file=f)
            print("Median time:", round(median(res_time_), 2), file=f)
            print("Min solvetime:", round(min(res_time_), 2), file=f)
            print("Max solvetime:", round(max(res_time_), 2), file=f)
            print("Variance of time:", round(variance(res_time_), 2), file=f)
            if tasklimit == total_tasks:
                print("Sigma:", round(math.sqrt(variance(res_time_)), 2), file=f)
                print(
                    "Real time for solving all tasks is ",
                    sum(res_time_),
                    sep="",
                    file=f,
                )
            else:
                print("Sd:", round(math.sqrt(variance(res_time_)), 2), file=f)
                print(
                    "Estimate time for solving all tasks is ",
                    avg_solvetime * total_tasks,
                    sep="",
                    file=f,
                )
            print("Average number of conflicts:", avg_conflicts, file=f)
            print(
                "Median number of conflicts:", round(median(res_conflicts_), 2), file=f
            )
            print("Min number of conflicts:", min(res_conflicts_), file=f)
            print("Max number of conflicts:", max(res_conflicts_), file=f)
            print(
                "Variance of number of conflicts:",
                round(variance(res_conflicts_), 2),
                file=f,
            )
            if tasklimit == total_tasks:
                print(
                    "Real number of conflicts for solving all tasks is ",
                    sum(res_conflicts_),
                    sep="",
                    file=f,
                )
            else:
                print(
                    "Estimate number of conflicts for solving all tasks is ",
                    avg_conflicts * total_tasks,
                    sep="",
                    file=f,
                )
            print("Number of SATs:", len(sats), file=f)
            if len(sats) > 0:
                print("SATs total runtime:", sum(sats), file=f)
                print("SATs average runtime:", mean(sats), file=f)
                print("SATs median runtime:", median(sats), file=f)
                print("SATs variance of runtime:", variance(sats), file=f)
            print("Number of UNSATs:", len(unsats), file=f)
            if len(unsats) > 0:
                print("UNSATs total runtime:", sum(unsats), file=f)
                print("UNSATs average runtime:", mean(unsats), file=f)
                print("UNSATs median runtime:", median(unsats), file=f)
                print("UNSATs variance of runtime:", variance(unsats), file=f)
    else:
        # Worker processes execute code below
        name = MPI.Get_processor_name()
        # print("I am a worker with rank %d on %s." % (rank, name))
        while True:
            comm.send(None, dest=0, tag=tags.READY)
            task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == tags.START:
                # Do the work here
                starttime = MPI.Wtime()
                results = []
                useless = False
                answer, time, conflicts = solve_vector(
                    task[0],
                    task[1],
                    input_decompose,
                    clauses,
                    max_var,
                    solver,
                    force_xor_flag,
                    force_disj_flag,
                )
                # if answer == 'UNSAT' and time < 1:
                # answer_useless, time_useless, conflicts_useless = solve_vector(task[0] ,task[1], input_decompose, template_clauses, max_var, solver, force_xor_flag, force_disj_flag)
                # if answer_useless == 'UNSAT':
                # useless = True
                comm.send(
                    tuple([task[1], time, conflicts, answer, useless]),
                    dest=0,
                    tag=tags.DONE,
                )
            elif tag == tags.EXIT:
                break
        comm.send(None, dest=0, tag=tags.EXIT)
