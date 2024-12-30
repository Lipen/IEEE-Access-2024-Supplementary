import argparse
import functools
import itertools
import math
import multiprocessing
import random
import subprocess
import sys
import time
from statistics import mean, variance

from ConfigSpace import Categorical, ConfigurationSpace, Constant
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from smac.facade.smac_ac_facade import SMAC4AC
from smac.scenario.scenario import Scenario
from pysat.formula import CNF

print = functools.partial(print, flush=True)


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--run-id",
        nargs="?",
        type=int,
        default=1,
        help="Set index of SMAC copy for pSMAC.",
    )
    parser.add_argument(
        "-cnf",
        "--cnfname",
        nargs="?",
        help="Path to CNF.",
    )
    parser.add_argument(
        "-np",
        "--numprocessors",
        type=int,
        default=36,
        help="Number of cores to multiprocessing.",
    )
    parser.add_argument(
        "-dt",
        "--dectype",
        nargs="?",
        type=str,
        default="id2",
        help="Lenght of chunks of input vars (id2 or id3).",
    )
    parser.add_argument(
        "-tlim",
        "--timelimit",
        nargs="?",
        type=int,
        default=84600,
        help="Total timelimit for SMAC.",
    )
    parser.add_argument(
        "-srlim",
        "--smacrunslimit",
        nargs="?",
        type=int,
        default=1000,
        help="Limit for SMAC runs (iterations).",
    )
    parser.add_argument(
        "-s",
        "--solver",
        nargs="?",
        type=str,
        default="kissat2022",
        help="Path to solver.",
    )
    parser.add_argument(
        "-stlim",
        "--subtasklimit",
        nargs="?",
        type=int,
        default=500,
        help="How many subtasks will be solved to construct estimate.",
    )
    parser.add_argument(
        "-sttlim",
        "--subtasktimelim",
        nargs="?",
        type=int,
        default=100,
        help="Timelimit for single subtask.",
    )
    return parser


def estimate(config):
    global numproc
    global solver
    global clauses
    global max_var
    global input_vars
    global miter_vars
    global subtasklimit
    global best_estimate
    global decomposition_type
    chunks, force_xor_flag = create_chunks(input_vars, decomposition_type)
    total_tasks = pow(2, len(chunks))
    if subtasklimit < total_tasks:
        tasks = [
            [nof_task, random_binary_vector(len(chunks))]
            for nof_task in range(subtasklimit)
        ]
    else:
        binary_vectors_ = list(itertools.product([0, 1], repeat=len(chunks)))
        tasks = [
            [nof_task, binary_vectors_[nof_task]]
            for nof_task in range(len(binary_vectors_))
        ]

    manager = multiprocessing.Manager()
    q = manager.Queue()
    p = multiprocessing.Pool(processes=numproc)
    jobs = []
    results_tabel = []
    for task in tasks:
        job = p.apply_async(
            solve_subtask,
            (
                task[0],
                task[1],
                chunks,
                clauses,
                max_var,
                solver,
                force_xor_flag,
                config,
            ),
        )
        jobs.append(job)
    for job in jobs:
        data = job.get()
        results_tabel.append(data)
    q.put("kill")
    p.close()
    p.join()
    indet_answers = [x[1] for x in results_tabel if x[1] == "INDET"]
    avg_time = mean([x[2] for x in results_tabel])
    avg_conf = mean([x[3] for x in results_tabel])
    print()
    print("Config: " + ", ".join(f"{key}={value}" for key, value in config.items()))
    print("Total subtasks:", total_tasks)
    print("Solved subtasks:", len(tasks))
    if len(indet_answers) > 0:
        print("Number of INDET answers between solved subtasks:", len(indet_answers))
    print("Average time for subtask:", round(avg_time, 3))
    print("Average conflicts for subtask:", round(avg_conf, 3))
    if subtasklimit == total_tasks:
        print("Sigma:", round(math.sqrt(variance([x[2] for x in results_tabel])), 3))
    else:
        print("Sd:", round(math.sqrt(variance([x[2] for x in results_tabel])), 3))
    time_estimate = avg_time * total_tasks
    conflicts_estimate = avg_conf * total_tasks
    print("Time estimate:", round(time_estimate, 3))
    print("Conflicts estimate:", round(conflicts_estimate, 3))
    if time_estimate < best_estimate:
        best_estimate = time_estimate
    return time_estimate


def create_chunks(input_vars: list[int], decompose_type: str):
    force_xor_flag = False
    if decompose_type[:2] == "id" or decompose_type[:2] == "bc":
        id_bc_flag = decompose_type[:2]
        decompose_type = decompose_type[2:]
    else:
        id_bc_flag = "id"
    if id_bc_flag == "id":
        chunks, force_xor_flag = create_id_chunks(input_vars, decompose_type)
    elif id_bc_flag == "bc":
        chunks = create_bc_vectors(decompose_type)
    return chunks, force_xor_flag


def create_id_chunks(input_vars, decompose_type):
    force_xor_flag = False
    nof_additional_chunks = 0
    if "_" in decompose_type:
        len_inp_chunks = [int(decompose_type.split("_")[0])]
        nof_additional_chunks = int(decompose_type.split("_")[1])
    elif "+" in decompose_type:
        len_inp_chunks = list(map(int, decompose_type.split("+")))
    elif "x" in decompose_type:
        len_inp_chunks = [int(decompose_type.split("x")[0])]
        force_xor_flag = True
    else:
        len_inp_chunks = [int(decompose_type)]

    # Create chunks
    input_decompose = []
    for i_ in len_inp_chunks:
        input_decompose_ = list(chunks(input_vars, i_))
        input_decompose_[-1] = input_vars[-i_:]
        input_decompose += input_decompose_

    # Additional chunks, if necessary
    i = 0
    while i < nof_additional_chunks:
        new_chunk = tuple(sorted(random.sample(input_vars, len_inp_chunks[0])))
        if new_chunk not in input_decompose:
            input_decompose.append(new_chunk)
            i += 1

    return input_decompose, force_xor_flag


def create_bc_vectors(decompose_type):
    global filename


def random_binary_vector(n):
    return tuple([random.randint(0, 1) for i in range(n)])


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield tuple(lst[i : i + n])


def solve_subtask(
    task_index, binaryvector, chunks, clauses, max_var, solver, force_xor_flag, config
):
    assert len(binaryvector) == len(chunks)
    new_clauses = []
    current_var = max_var
    for bit, inp_chunk in zip(binaryvector, chunks):
        if len(inp_chunk) == 2:
            new_clauses_, current_var = encode_2XOR_clauses(inp_chunk, current_var)
        elif len(inp_chunk) == 3:
            if force_xor_flag == True:
                new_clauses_, current_var = encode_3XOR_clauses(inp_chunk, current_var)
            else:
                new_clauses_, current_var = encode_MAJORITY_clauses(
                    inp_chunk, current_var
                )
        new_clauses.extend(new_clauses_)
        new_clauses.append([current_var] if bit == 1 else [-current_var])
    answer, time, conflicts = create_and_solve_CNF(
        task_index, clauses, new_clauses, current_var, solver, config
    )
    return task_index, answer, time, conflicts


def encode_2XOR_clauses(inp_pair, current_var):
    current_var += 1
    clauses = [
        [current_var, -inp_pair[0], inp_pair[1]],
        [current_var, inp_pair[0], -inp_pair[1]],
        [-current_var, inp_pair[0], inp_pair[1]],
        [-current_var, -inp_pair[0], -inp_pair[1]],
    ]
    return clauses, current_var


def encode_3XOR_clauses(inp_chunk, current_var):
    current_var += 1
    clauses = [
        [-current_var, inp_chunk[0], inp_chunk[1], inp_chunk[2]],
        [current_var, inp_chunk[0], inp_chunk[1], -inp_chunk[2]],
        [current_var, inp_chunk[0], -inp_chunk[1], inp_chunk[2]],
        [-current_var, inp_chunk[0], -inp_chunk[1], -inp_chunk[2]],
        [current_var, -inp_chunk[0], inp_chunk[1], inp_chunk[2]],
        [-current_var, -inp_chunk[0], inp_chunk[1], -inp_chunk[2]],
        [-current_var, -inp_chunk[0], -inp_chunk[1], inp_chunk[2]],
        [current_var, -inp_chunk[0], -inp_chunk[1], -inp_chunk[2]],
    ]
    return clauses, current_var


def encode_MAJORITY_clauses(inp_triple, current_var):
    current_var += 1
    clauses = [
        [-current_var, inp_triple[0], inp_triple[1], inp_triple[2]],
        [-current_var, inp_triple[0], inp_triple[1], -inp_triple[2]],
        [-current_var, inp_triple[0], -inp_triple[1], inp_triple[2]],
        [current_var, inp_triple[0], -inp_triple[1], -inp_triple[2]],
        [-current_var, -inp_triple[0], inp_triple[1], inp_triple[2]],
        [current_var, -inp_triple[0], inp_triple[1], -inp_triple[2]],
        [current_var, -inp_triple[0], -inp_triple[1], inp_triple[2]],
        [current_var, -inp_triple[0], -inp_triple[1], -inp_triple[2]],
    ]
    return clauses, current_var


def create_and_solve_CNF(task_index, clauses, new_clauses, max_var, solver, config):
    cnf_str = create_str_CNF(clauses, new_clauses, max_var)
    result = solve_CNF(cnf_str, solver, config)
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
    if answer == "INDET":
        time = time * 10
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


def solve_CNF(cnf, solver, config):
    args = [solver]
    for key, value in config.items():
        args.append(f"--{key}={value}")
    solver = subprocess.run(args, capture_output=True, text=True, input=cnf)
    result = solver.stdout.split("\n")
    errors = solver.stderr
    # print(result)
    if errors:
        print(errors)
    return result


def run_smac(scenario, run_id):
    smac = SMAC4AC(scenario=scenario, tae_runner=estimate, run_id=run_id)
    best_found_config = smac.optimize()
    return [run_id, best_estimate, best_found_config]


##################################################
# ----------------------MAIN-------------------- #
##################################################


if __name__ == "__main__":
    time_start = time.time()
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    run_id = namespace.run_id
    filename = namespace.cnfname
    numproc = namespace.numprocessors
    solver = namespace.solver
    timelimit = namespace.timelimit
    smacrunslimit = namespace.smacrunslimit
    subtasklimit = namespace.subtasklimit
    decomposition_type = namespace.dectype
    subtasktimelimit = namespace.subtasktimelim
    best_estimate = math.inf
    instance_name = "".join(filename.split("/")[-1].split(".")[:-1])
    INT_MAX = 2**31 - 1

    cnf = CNF(from_file=filename)
    clauses = cnf.clauses
    max_var = cnf.nv
    input_vars = list(
        map(
            int,
            [comment for comment in cnf.comments if "c inputs: " in comment][0]
            .split(":")[1]
            .split(),
        )
    )
    miter_vars = list(
        map(
            int,
            [comment for comment in cnf.comments if "c miter variables: " in comment][0]
            .split(":")[1]
            .split(),
        )
    )

    # 'ands' -- extract and eliminate and gates
    # 'backbone' -- binary clause backbone (2=eager)
    # 'backboneeffort' -- effort in per mille
    # 'backbonemaxrounds' -- maximum backbone rounds
    # 'backbonerounds' -- backbone rounds limit
    # 'bump' -- enable variable bumping
    # 'bumpreasons' -- bump reason side literals too
    # 'bumpreasonslimit' -- relative reason literals limit
    # 'bumpreasonsrate' -- decision rate limit
    # 'chrono' -- allow chronological backtracking
    # 'chronolevels' -- maximum jumped over levels
    # 'compact' -- enable compacting garbage collection
    # 'compactlim' -- compact inactive limit (in percent)
    # 'decay' -- per mille scores decay
    # 'definitioncores' -- how many cores
    # 'definitions' -- extract general definitions
    # 'definitionticks' -- kitten ticks limits
    # 'defraglim' -- usable defragmentation limit in percent
    # 'defragsize' -- size defragmentation limit
    # 'eliminate' -- bounded variable elimination (BVE)
    # 'eliminatebound' -- maximum elimination bound
    # 'eliminateclslim' -- elimination clause size limit
    # 'eliminateeffort' -- effort in per mille
    # 'eliminateinit' -- initial elimination interval
    # 'eliminateint' -- base elimination interval
    # 'eliminateocclim' -- elimination occurrence limit
    # 'eliminaterounds' -- elimination rounds limit
    # 'emafast' -- fast exponential moving average window
    # 'emaslow' -- slow exponential moving average window
    # 'equivalences' -- extract and eliminate equivalence gates
    # 'extract' -- extract gates in variable elimination
    # 'forcephase' -- force initial phase
    # 'forward' -- forward subsumption in BVE
    # 'forwardeffort' -- effort in per mille
    # 'ifthenelse' -- extract and eliminate if-then-else gates
    # 'incremental' -- enable incremental solving
    # 'mineffort' -- minimum absolute effort in millions
    # 'minimize' -- learned clause minimization
    # 'minimizedepth' -- minimization depth
    # 'minimizeticks' -- count ticks in minimize and shrink
    # 'modeinit' -- initial focused conflicts limit
    # 'otfs' -- on-the-fly strengthening
    # 'phase' -- initial decision phase
    # 'phasesaving' -- enable phase saving
    # 'probe' -- enable probing
    # 'probeinit' -- initial probing interval
    # 'probeint' -- probing interval
    # 'profile' -- profile level
    # 'promote' -- promote clauses
    # 'reduce' -- learned clause reduction
    # 'reducefraction' -- reduce fraction in percent
    # 'reduceinit' -- initial reduce interval
    # 'reduceint' -- base reduce interval
    # 'reluctant' -- stable reluctant doubling restarting
    # 'reluctantint' -- reluctant interval
    # 'reluctantlim' -- reluctant limit (0=unlimited)
    # 'rephase' -- reinitialization of decision phases
    # 'rephaseinit' -- initial rephase interval
    # 'rephaseint' -- base rephase interval
    # 'restart' -- enable restarts
    # 'restartint' -- base restart interval
    # 'restartmargin' -- fast/slow margin in percent
    # 'shrink' -- learned clauses (1=bin, 2=lrg, 3=rec)
    # 'simplify' -- enable probing and elimination
    # 'stable' -- enable stable search mode
    # 'substitute' -- equivalent literal substitution
    # 'substituteeffort' -- effort in per mille
    # 'substituterounds' -- maximum substitution rounds
    # 'subsumeclslim' -- subsumption clause size limit
    # 'subsumeocclim' -- subsumption occurrence limit
    # 'sweep' -- enable SAT sweeping
    # 'sweepclauses' -- environment clauses
    # 'sweepdepth' -- environment depth
    # 'sweepeffort' -- effort in per mille
    # 'sweepfliprounds' -- flipping rounds
    # 'sweepmaxclauses' -- maximum environment clauses
    # 'sweepmaxdepth' -- maximum environment depth
    # 'sweepmaxvars' -- maximum environment variables
    # 'sweepvars' -- environment variables
    # 'target' -- target phases (1=stable, 2=focused)
    # 'tier1' -- learned clause tier one glue limit
    # 'tier2' -- learned lause tier two glue limit
    # 'tumble' -- tumbled external indices order
    # 'vivify' -- vivify clauses
    # 'vivifyeffort' -- effort in per mille
    # 'vivifyirred' -- relative irredundant effort
    # 'vivifytier1' -- relative tier1 effort
    # 'vivifytier2' -- relative tier2 effort
    # 'walkeffort' -- effort in per mille
    # 'walkinitially' -- initial local search
    # 'warmup' -- initialize phases by unit propagation

    config = ConfigurationSpace()

    # 'time' -- timelimit
    config.add_hyperparameter(Constant("time", subtasktimelimit))

    # fmt: off
    config.add_hyperparameter(UniformIntegerHyperparameter(name="ands", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="backbone", lower=0, upper=2, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="bump", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="bumpreasons", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="chrono", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="compact", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(Categorical(name="decay", items=[1, 4, 10, 16, 50, 64, 200], default=50))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="definitions", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="eliminate", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(Categorical(name="eliminatebound", items=[1, 4, 10, 16, 64, 256, 1024, 4096, 8192], default=16))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="equivalences", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="extract", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="forcephase", lower=0, upper=1, default_value=0))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="forward", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="ifthenelse", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="minimize", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(Categorical(name="minimizedepth", items=[1, 4, 10, 16, 64, 256, 1000, 4096, 16384, 65536, 262144, 1000000], default=1000))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="minimizeticks", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="otfs", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="phase", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="phasesaving", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="probe", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="promote", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="reduce", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(Categorical(name="reducefraction", items=[10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100], default=75))
    config.add_hyperparameter(Categorical(name="reduceinit", items=[2, 4, 10, 16, 64, 256, 1000, 4096, 16384, 65536, 100000], default=1000))
    config.add_hyperparameter(Categorical(name="reduceint", items=[2, 4, 10, 16, 64, 256, 1000, 4096, 16384, 65536, 100000], default=1000))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="reluctant", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="rephase", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(Categorical(name="restartint", items=[1, 4, 10, 16, 64, 256, 1000, 4096, 10000], default=1000))
    config.add_hyperparameter(Categorical(name="restartmargin", items=[0, 5, 10, 15, 20, 25], default=10))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="shrink", lower=0, upper=3, default_value=3))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="simplify", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="stable", lower=0, upper=2, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="substitute", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="sweep", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="tumble", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="vivify", lower=0, upper=1, default_value=1))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="walkinitially", lower=0, upper=1, default_value=0))
    config.add_hyperparameter(UniformIntegerHyperparameter(name="warmup", lower=0, upper=1, default_value=1))
    # fmt: on

    smac_workdir = f"smacrun_{len(config)-1}params_{subtasktimelimit}sectimelim_{subtasklimit}subtasks_{instance_name}"
    scenario = Scenario(
        {
            "run_obj": "quality",
            "cs": config,
            "wallclock_limit": timelimit,
            "runcount-limit": smacrunslimit,
            "shared_model": True,
            "output_dir": smac_workdir,
            "input_psmac_dirs": smac_workdir,
            "deterministic": True,
        }
    )

    result = run_smac(scenario, run_id)
    best_value = result[1]
    best_config = result[2]
    best_config_line = " ".join(
        f"--{key}={value}" for key, value in best_config.items()
    )
    best_config_comments = [f"c --{key}={value}" for key, value in best_config.items()]
    print()
    print(f"Run id: {run_id}")
    print(f"Best value: {best_value}")
    print(f"Best config: {best_config_line}")
    print()
    print(f"All done in {time.time() - time_start:.3f}s!")
