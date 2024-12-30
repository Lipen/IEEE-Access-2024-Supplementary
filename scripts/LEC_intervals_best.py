import sys
import argparse
import functools
import subprocess
import math
from itertools import combinations, product
from statistics import mean, median, variance

from mpi4py import MPI
from pysat.pb import *

print = functools.partial(print, flush=True)


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", nargs="?", default="lec_CvK_12.cnf")
    parser.add_argument("-bp", "--bestparams", nargs="?", default="CvK_12_10")
    parser.add_argument("-lim", "--limit", nargs="?", type=int, default=100)
    parser.add_argument("-e", "--encoding_type", nargs="?", type=int, default=1)
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
                vars_right = [x for x in range(int(dimacs[i].split()[-3]), int(dimacs[i].split()[-1]) + 1)]
            elif "c outputs first scheme" in dimacs[i]:
                outputs_first = list(map(int, dimacs[i].split(":")[1].split()))
            elif "c variables for gates in second scheme" in dimacs[i]:
                vars_left = [x for x in range(int(dimacs[i].split()[-3]), int(dimacs[i].split()[-1]) + 1)]
            elif "c outputs second scheme" in dimacs[i]:
                outputs_second = list(map(int, dimacs[i].split(":")[1].split()))
            elif "c miter variables" in dimacs[i]:
                miter_vars = list(map(int, dimacs[i].split(":")[1].split()))
            elif "c cubes variables:" in dimacs[i]:
                cubes_vars = list(map(int, dimacs[i].split(":")[1].split()))
            elif "c outputs: " in dimacs[i]:
                outputs = list(map(int, dimacs[i].split(":")[1].split()))
            elif "c variables for gates:" in dimacs[i]:
                gates_vars = [x for x in range(int(dimacs[i].split()[-3]), int(dimacs[i].split()[-1]) + 1)]
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


def solve_range(
    task_index,
    current_range,
    clauses,
    input_vars,
    weights,
    max_var,
    solver,
    encoding_type,
    best_params,
):
    lower_bound = current_range[0]
    upper_bound = current_range[-1]
    cnf_lower = PBEnc.geq(input_vars, weights, bound=lower_bound, top_id=max_var, encoding=encoding_type)
    cnf_upper = PBEnc.leq(
        input_vars,
        weights,
        bound=upper_bound,
        top_id=max([max_var, cnf_lower.nv]),
        encoding=encoding_type,
    )
    new_max_var = max([max_var, cnf_lower.nv, cnf_upper.nv])
    new_clauses = cnf_lower.clauses + cnf_upper.clauses
    answer, solvetime, conflicts = create_and_solve_CNF(
        task_index, clauses, new_clauses, new_max_var, solver, best_params
    )
    return answer, solvetime, conflicts


def create_and_solve_CNF(task_index, clauses, new_clauses, max_var, solver, best_params):
    cnf_str = create_str_CNF(clauses, new_clauses, max_var)
    # with open('lec_CvK_16_ld4_rand_subtask' + str(task_index) + '.cnf', 'w') as f:
    # print(cnf_str, file = f)
    result = solve_CNF(cnf_str, solver, best_params)
    answer = "INDET"
    solvetime = None
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
            solvetime = float(line.split()[-2])
        elif ("c conflicts:" in line and "kissat" in solver) or ("c conflicts:" in line and "cadical" in solver):
            conflicts = int(line.split()[-4])
        elif "c conflicts " in line:
            conflicts = int(line.split()[3])
    # if task_index < 180 and answer == 'SAT':
    #  with open('a5_1_64_11_d3_t' + str(task_index) + '.cnf', 'w') as f:
    #    print(cnf_str, file = f)
    return answer, solvetime, conflicts


def create_str_CNF(clauses, new_clauses, max_var):
    lines = []
    header_ = "p cnf " + str(max_var) + " " + str(len(clauses) + len(new_clauses))
    lines.append(header_)
    lines.extend([" ".join(list(map(str, clause))) + " 0" for clause in clauses])
    lines.extend([" ".join(list(map(str, clause))) + " 0" for clause in new_clauses])
    cnf = "\n".join(lines)
    # pprint.pprint(cnf)
    return cnf


def solve_CNF(cnf, solver, best_params):
    config = [solver]
    config += best_params
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


def make_ranges(input_vars, nof_ranges):
    l = range(2 ** len(input_vars))
    n = len(l)
    k = nof_ranges
    return [l[i * (n // k) + min(i, n % k) : (i + 1) * (n // k) + min(i + 1, n % k)] for i in range(k)]


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
        cnf_name = "".join(filename.split("/")[-1].split(".")[:-1])
        nof_ranges = namespace.limit
        solver = namespace.solver
        encoding_type = namespace.encoding_type
        best_params_file = namespace.bestparams
        best_params_str = open(best_params_file, "r").readlines()[0]
        best_params = best_params_str[:-1].split() if best_params_str[-1] == "\n" else best_params_str.split()
        bp_sample = int(best_params_file.split("_")[-1])

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

        weights = [2**x for x in reversed(range(len(inputs)))]
        ranges = make_ranges(inputs, nof_ranges)

    else:
        solver = None
        max_var = None
        clauses = None
        inputs = None
        encoding_type = None
        weights = None
        best_params = None

    solver = comm.bcast(solver, root=0)
    clauses = comm.bcast(clauses, root=0)
    max_var = comm.bcast(max_var, root=0)
    inputs = comm.bcast(inputs, root=0)
    encoding_type = comm.bcast(encoding_type, root=0)
    weights = comm.bcast(weights, root=0)
    best_params = comm.bcast(best_params, root=0)

    if rank == 0:
        # Master process executes code below
        task_index = 0
        num_workers = size - 1
        sats = []
        unsats = []
        closed_workers = 0
        results_table = []
        filename = (
            "tmp_log_ID_PBR_"
            + "_"
            + cnf_name
            + "_ranges"
            + str(nof_ranges)
            + "_sample"
            + str(bp_sample)
            + "_et"
            + str(encoding_type)
            + "_"
            + solver
            + ".log"
        )
        with open(filename, "w") as f:
            print(
                "Master starting with %d workers and %d tasks" % (num_workers, nof_ranges),
                file=f,
            )
            print("Inputs:", inputs, file=f)
            print("Weights:", weights, file=f)
            print("Ranges:", ranges, file=f)
            while closed_workers < num_workers:
                data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                source = status.Get_source()
                tag = status.Get_tag()
                if tag == tags.READY:
                    # Worker is ready, so send it a task
                    if task_index < nof_ranges:
                        print("", file=f)
                        task = (task_index, ranges[task_index])
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
                        "Got data from worker %d (current runtime % 6.2f)" % (source, (MPI.Wtime() - start_time)),
                        file=f,
                    )
                    print(data, file=f)
                    results_table.append(data)
                    if "UNSAT" in data[3]:
                        unsats.append(data[1])
                    elif "SAT" in data[3]:
                        sats.append(data[1])
                    current_avg_solvetime = round(sum([x[1] for x in results_table]) / len(results_table), 2)
                    current_avg_conflicts = round(sum([x[2] for x in results_table]) / len(results_table), 2)
                    print(
                        "Current time estimate: ",
                        current_avg_solvetime * nof_ranges,
                        "Current ratio:",
                        (current_avg_conflicts * nof_ranges) / pow(2, len(inputs)),
                        file=f,
                    )
                elif tag == tags.EXIT:
                    print("Worker %d exited." % source, file=f)
                    closed_workers += 1
            print("Master finishing", file=f)
            print("Total runtime:", MPI.Wtime() - start_time, "on", size, "cores", file=f)
            print("", file=f)
            res_time_ = [x[1] for x in results_table]
            avg_solvetime = round(sum(res_time_) / len(results_table), 2)
            res_conflicts_ = [x[2] for x in results_table]
            avg_conflicts = round(sum(res_conflicts_) / len(results_table), 2)
            print("CNF name:", cnf_name, file=f)
            print("Solver:", solver, file=f)
            print("Decomposition type: PBR_bdd", file=f)
            print("Solved (all)", len(results_table), "tasks.", file=f)
            print("Total tasks:", nof_ranges, file=f)
            print("Average solvetime:", avg_solvetime, file=f)
            print("Median time:", round(median(res_time_), 2), file=f)
            print("Min solvetime:", round(min(res_time_), 2), file=f)
            print("Max solvetime:", round(max(res_time_), 2), file=f)
            print("Variance of time:", round(variance(res_time_), 2), file=f)
            if len(results_table) == nof_ranges:
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
                    avg_solvetime * nof_ranges,
                    sep="",
                    file=f,
                )
                print()
            print("Average number of conflicts:", avg_conflicts, file=f)
            print("Median number of conflicts:", round(median(res_conflicts_), 2), file=f)
            print("Min number of conflicts:", min(res_conflicts_), file=f)
            print("Max number of conflicts:", max(res_conflicts_), file=f)
            print(
                "Variance of number of conflicts:",
                round(variance(res_conflicts_), 2),
                file=f,
            )
            if len(results_table) == nof_ranges:
                print(
                    "Real total number of conflicts for solving all tasks is ",
                    sum(res_conflicts_),
                    sep="",
                    file=f,
                )
                print(
                    "(Number of conflicts / Brutforce actions) ratio:",
                    round(sum(res_conflicts_) / pow(2, len(inputs)), 10),
                    file=f,
                )
            else:
                print(
                    "Estimate number of conflicts for solving all tasks is ",
                    avg_conflicts * nof_ranges,
                    sep="",
                    file=f,
                )
                print(
                    "(Number of conflicts / Brutforce actions) ratio:",
                    round((avg_conflicts * nof_ranges) / pow(2, len(inputs)), 10),
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
                answer, solvetime, conflicts = solve_range(
                    task[0],
                    task[1],
                    clauses,
                    inputs,
                    weights,
                    max_var,
                    solver,
                    encoding_type,
                    best_params,
                )
                comm.send(
                    tuple([task[1], solvetime, conflicts, answer]),
                    dest=0,
                    tag=tags.DONE,
                )
            elif tag == tags.EXIT:
                break
        comm.send(None, dest=0, tag=tags.EXIT)
