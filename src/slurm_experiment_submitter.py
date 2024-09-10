#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    The logic is as follows:
        submit_on_cluster should be True on the cluster and False otherwise, for debugging.
        num_cpus is the number of cpus to ask from the cluster, and number of workers to pass to python.
        cluster sbatch directives go in cluster_text.
        All params that should be shared across all commands go in default_params.
        num_seeds is a global param, the number of seeds for each experiment (running in parallel).
        experiment_list_env is a list of strings of 1 environment each. Benchmark can take more than 1 env, but the
        configuration that sets up sbatch id to path cannot. Run num_envs (1) * num_seeds (N) in parallel.
            Should match length of experiment_list_params.
        experiment_list_params are the params to run each individual experiment with (experiment = one param config,
            1 environment, >= 1 seeds).
"""

import os
import random
import subprocess
import argparse
import time


def save_to_file(strings, filename):
    with open(filename, 'w') as file:
        for string in strings:
            file.write(string + '\n')

def make_sbatch_script(environment, seed=0, run_name=None, results_path=None, runtime=1.0,
                       qos="medium", exploration_beta=0.0, learning_rate=0.001, sample_action=True,
                       sample_action_from_improved_policy=False,
                       exploitation_beta=0.0, maximum_number_of_iterations=10,
                       epistemic_exploration_in_selfplay=None, rescale_q=True,
                       ):
    jobname = "eaz" + "_" + environment.replace('minatar-', '')
    if runtime < 1:
        runtime = f"01:00:00"
    if runtime < 10:
        runtime = f"0{int(runtime)}:00:00"
    elif runtime < 100:
        runtime = f"{int(runtime)}:00:00"
    else:
        raise ValueError("runtime must be < 100 hours for HPC")

    full_params = f"seed={seed} env_id={environment} wandb_run_name='{run_name}' exploration_beta={exploration_beta} " \
                  f"learning_rate={learning_rate} " \
                  f"sample_actions={sample_action} " \
                  f"sample_from_improved_policy={sample_action_from_improved_policy} " \
                  f"exploitation_beta={exploitation_beta} " \
                  f"maximum_number_of_iterations={maximum_number_of_iterations} " \
                  f"directed_exploration={epistemic_exploration_in_selfplay} " \
                  f"rescale_q_values_in_search={rescale_q}"

    script_text = f"""#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --time={runtime}
#SBATCH --qos={qos}
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
##SBATCH --partition=gpu
#SBATCH --mem-per-cpu=4GB
#SBATCH --partition=st,general,insy

# Setup env variables
export WANDB_CACHE_DIR="/tudelft.net/staff-umbrella/inadequate/emctx/wandb/cache/"
export APPTAINERENV_WANDB_API_KEY=
export APPTAINERENV_WANDB_DIR="/mnt/umbrella_emctx/"

hostname
/usr/bin/nvidia-smi

previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

# apptainer command
srun apptainer exec --nv --mount type=bind,src=/tudelft.net/staff-umbrella/inadequate/emctx/,dst=/mnt/umbrella_emctx/ --mount type=bind,src=/tudelft.net/staff-umbrella/inadequate/emctx/e-alphazero/src/,dst=/mnt/alphazero/ /tudelft.net/staff-umbrella/inadequate/emctx/container/emctx_container.sif python3 /mnt/alphazero/main.py {full_params}

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"""
    return script_text, full_params

def make_all_experiments(num_seeds, exploration_betas, environments, learning_rates,
                         results_path, sample_actions, sample_actions_from_improved_policy, exploitation_betas,
                         maximum_number_of_iterations, epistemic_exploration_in_selfplay, rescale_qs):
    """
        This function returns a list of all sbatch files with the right hps (env, sigma, n, k) x 2.
        The first is the real experiment: viac, env, sigma, n, k),
        the second is the baseline: env, sigma=1, n=k, k

        returns a list of all experiment

        agents_list: list(str)
            The file names without .py (assumed to be in the same dir, in the path specified in make_sbatch_script)
        environments_list: list(str)
            The list of envs to run experiments on
        total_timesteps_list: list(int)
            A list of the same length as environments_list, of the total training steps per env. In ints or xey.
        sigma: float
            The factor by which the STD of the normal of SAC / TD3 is multiplied, for the VIAC update.
        n: int
            The number of proposed actions.
        k: int
            The number of top-k action-values averaged over (the "improved policy").
    """
    all_experiments = []
    seeds = []
    run_names = []
    full_params_list = []
    paths_list = []
    low = 0
    high = 10000000

    for env_id in environments:
        for exploration_beta in exploration_betas:
            for learning_rate in learning_rates:
                for sample_action in sample_actions:
                    for sample_action_from_improved_policy in sample_actions_from_improved_policy:
                        for exploitation_beta in exploitation_betas:
                            for epistemic_exploration in epistemic_exploration_in_selfplay:
                                for rescale_q in rescale_qs:
                                    for _ in range(num_seeds):
                                        # Compute runtime
                                        runtime = maximum_number_of_iterations / 100  # td3 is about 1:10 hours per 1m steps, so this is in hours
                                        if runtime <= 4:
                                            qos = "short"
                                        elif runtime <= 24:
                                            qos = "medium"
                                        else:
                                            qos = "long"
                                        # Setup viac experiments
                                        seed = random.randrange(low, high)
                                        run_name = f"eaz_{seed}_{env_id}_{time.asctime(time.localtime(time.time()))}"
                                        local_results_path = results_path + f"/{env_id}/eaz/"
                                        script_text, full_params = make_sbatch_script(environment=env_id,
                                                                                      seed=seed,
                                                                                      run_name=run_name,
                                                                                      results_path=local_results_path,
                                                                                      runtime=runtime,
                                                                                      qos=qos,
                                                                                      exploration_beta=exploration_beta,
                                                                                      learning_rate=learning_rate,
                                                                                      sample_action=sample_action,
                                                                                      sample_action_from_improved_policy=sample_action_from_improved_policy,
                                                                                      exploitation_beta=exploitation_beta,
                                                                                      maximum_number_of_iterations=maximum_number_of_iterations,
                                                                                      epistemic_exploration_in_selfplay=epistemic_exploration,
                                                                                      rescale_q=rescale_q,
                                                                                      )
                                        seeds.append(seed)
                                        run_names.append(run_name)
                                        paths_list.append(f"/{env_id}/eaz/{run_name}")
                                        all_experiments.append(script_text)
                                        full_params_list.append(full_params)

    return all_experiments, run_names, seeds, full_params_list, paths_list


submit_on_cluster = False
environments = ["minatar-breakout", "minatar-asterix", "minatar-seaquest", "minatar-freeway"]
num_seeds = 2
purpose = "Tuning exploration variations of EAZ on minatar"
results_path = "/mnt/results"  # "/home/yaniv"      # "/tudelft.net/staff-umbrella/yaniv/viac/results"
true_results_path = "/tudelft.net/staff-umbrella/inadequate/emctx/results"
local_results_path = "/home/yaniv"
maximum_number_of_iterations = 500
exploration_betas = [20.0]
exploitation_betas = [0.0]
learning_rates = [0.001] # [0.005, 0.001, 0.5 * 0.001, 0.0001, 0.5 * 0.0001]
sample_actions = [True, False]
sample_actions_from_improved_policy = [False]
scale_values = [False]
epistemic_exploration_in_selfplay = [False, True]
save_jobs_paths = True
job_paths = []
job_paths_file_name = f"{purpose} {time.asctime(time.localtime(time.time()))}"

all_experiments, run_names, seeds, full_params_list, paths_list = make_all_experiments(num_seeds=num_seeds,
                                                                                       exploration_betas=exploration_betas,
                                                                                       environments=environments,
                                                                                       learning_rates=learning_rates,
                                                                                       results_path=results_path,
                                                                                       sample_actions=sample_actions,
                                                                                       sample_actions_from_improved_policy=sample_actions_from_improved_policy,
                                                                                       exploitation_betas=exploitation_betas,
                                                                                       maximum_number_of_iterations=maximum_number_of_iterations,
                                                                                       epistemic_exploration_in_selfplay=epistemic_exploration_in_selfplay,
                                                                                       rescale_qs=scale_values)

first_job_id = 0
last_job_id = 0

for index, experiment_file_text in enumerate(all_experiments):
    sbatch_file_name = './temporary_eaz_runner.sh'
    with open(sbatch_file_name, "w") as file:
        file.write(experiment_file_text)
    if submit_on_cluster:
        process = subprocess.Popen(f"sbatch {sbatch_file_name}", shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        experiment_results_path = true_results_path + paths_list[index]
        # Check if the job submission was successful
        if process.returncode == 0:
            # The job ID is usually in the first line of the stdout
            job_id = stdout.decode().strip().split()[-1]
            if index == 0:
                first_job_id = job_id
            else:
                last_job_id = job_id
            print(f"{job_id}\n"
                  f"{experiment_results_path}\n"
                  f"{full_params_list[index]} "
                  , flush=True)
            job_paths.append(experiment_results_path)
        else:
            print(f"Job submission failed. Error message:\n{stderr.decode()}")
    else:
        experiment_results_path = true_results_path + paths_list[index]
        job_paths.append(experiment_results_path)
        print(f"{experiment_results_path}\n"
              f"{full_params_list[index]}")
        print(experiment_file_text)

print(f"Experiment summary: \n"
      f"Purpose: {purpose} \n"
      f"num_seeds = {num_seeds}\n"
      f"environments = {environments}\n"
      f"maximum_number_of_iterations = {maximum_number_of_iterations}\n"
      f"exploration_betas = {exploration_betas}\n"
      f"exploitation_betas = {exploitation_betas}\n"
      f"learning_rates = {learning_rates}\n"
      f"sample_actions = {sample_actions}\n"
      f"sample_actions_from_improved_policy = {sample_actions_from_improved_policy}\n"
      f"scale_values = {scale_values}\n"
      f"Saving the job - names to file: {save_jobs_paths} \n"
      f"For a total number of jobs: {len(all_experiments)} \n"
      f"Jod ids: {first_job_id} to {last_job_id} \n"
      )

if save_jobs_paths:
    if submit_on_cluster:
        experiment_names_file_paths = true_results_path + "/" + job_paths_file_name + ".txt"
        save_to_file(job_paths, experiment_names_file_paths)
        print(f"Saved file {job_paths_file_name} to path: {experiment_names_file_paths}")
    else:
        experiment_names_file_paths = local_results_path + "/" + job_paths_file_name + ".txt"
        save_to_file(job_paths, experiment_names_file_paths)
        print(f"Saved file {job_paths_file_name} to path: {experiment_names_file_paths}")