# TODO

- [x] Selfplay
  - [x] Start with pgx example
  - [x] Modify to save states and exploration policy
  - [x] Modify to use exploration policy during selfplay
  - [x] Use uncertainty
- [~] Network
  - [x] Modify network to have two policy heads (exploration / exploitation)
  - [x] Output reward and value uncertainty (local / UBE)
  - [x] UBE computed for AZ as max(scaled_reward_unc * geometric_series_coeff, u)
  - [x] reward uncertainty scaled correctly to scaled_reward_unc
  - [ ] Maybe change to transformer
- [x] Add states to replay buffer
- [ ] Reanalyze
  - [x] Prioritized sampling from replay buffer
  - [x] Redo search to calculate targets
      - [x] Value targets
      - [x] Exploitation Policy
      - [x] UBE/Uncertainty?
- [x] Training
  - [x] Take samples from reanalyze
  - [x] Calculate gradients and back-propagate
- [x] Evaluation
  - [x] Start with pgx example
  - [x] Modify to use exploitation policy
  - [x] Add search?
- [x] Local uncertainty
  - [x] implement hashes
  - [x] add local uncertainty to network
  - [x] use local uncertainty in code
- [x] Pretraining: Start training after X random moves
- [x] Add auto-debug-params-loading
- [x] Setup correct default params for minatar
- [~] Logging
  - [x] Root UBE
  - [x] Root value
  - [x] Raw value
  - [x] Raw UBE
  - [ ] Replay buffer uniqueness
- [x] Pessimistic execution
- [x] UBE clipped < max_uncertainty which is a HP and > 0
- [x] automated beta scheduling
- [x] EpistemicQ transform instead of Q transform
- [x] Func. to choose between exploration / exploitation logits in selfplay
- [x] Correct ube / reward uncertainty computations
- [x] HP for sampling actions from root / using selected action det.
- [x] HP for sampling actions from improved policy instead of visit counts
- [x] Train / interact ratio >= 2
- [x] HP to choose whether to rescale the qs 0-1 or not, for gumbel search
- [x] Count number of hashed states
- [x] Selfplay continuous from where it left off at the last iteration.
- [x] Verify root-std behavior.
- [x] Add functionality to check if game = 2 player, and change the discount to * -1
- [x] Correct exploration policy targets to completed_q_variances

- [ ] Change environment to jax_jit env
- [ ] Fix the LCG hash
- [ ] Tuning




# Subleq

- Different agent per problem
- Start with reward = 1 for solving it, 0 for not

Scale of the game = N

State:
    - Input: one-hot encoded vector of fixed size K   [N x K]
    - Output: -||- [N x K]
    - Memory before executing (program): one-hot encoded [N x N]

    - Input after executing: one hot encoded vector of fixed size K   [N x K]
    - Output after executing: -||- [N x K]
    - Memory after executing: [N x N]

    - size = 2 (2nk + n*n) = 4nk + 2n^2

Actions:
    - 1 action per possible byte [1 x N]

Environment:
    - Predefined input/output for the specific problem
    - After each action, execute the code on the predefined input
        - If there is a wrong output -> immediate termination -> 0 reward
        - If it terminates correctly (all outputs matched), test all on all other input sequences -> reward
        - Otherwise, 0 reward and game continues
    - Next observation is the initial state of memory and input/output + state of memory and input/output after executing the code

Architecture:

- "Just an MLP"
- fully connected, hidden layers ("256 is a good number")

Questions:
1. Do we actually want to include input/output (yes), after-execution stuff (input/output at least) in the observation?
  Observation: current memory state (the instructions it has written) + [test input, desired output, remaining input after execution, output from execution] * test_case_num
2. If yes, do we want to have the same input/output each time or a random test case (per episode)?
  Same input/output per task, because it helps with hashing
3. If yes, do we want to have multiple test cases in the observation?
  Maybe? make it a hyperparam / or at least easy to add in the future
4. Should reward be determined by test case or all test cases?
  Reward function per task?

