# Neuro-Symbolic Inverse Constrained Reinforcement Learning

This repository contains the official implementation of the paper **"Neuro-Symbolic Inverse Constrained Reinforcement Learning"**. It integrates symbolic reasoning with inverse reinforcement learning under logical constraints. The approach includes policy learning, symbolic generalization, and constraint deduction using logic-based methods like ACUITY.

---

## Getting Started

### Prerequisites

- **Python 3.x**
- **SWI-Prolog (swipl)**

To install SWI-Prolog, follow the instructions [here](https://www.swi-prolog.org/Download.html).

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/nsicrl.git
    cd nsicrl
    ```

2. (Optional) Create a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install Python dependencies:

    ```bash
    pip install -r requirements.txt
    ```

---

## Running the Code

Navigate to the `src/` directory and run the main script:

```bash
cd src/
python main.py
```

All arguments are set to default values and the code should run out of the box.

---

## Output

- Figures will be saved in the `figures/` directory.
- Output includes:
  - A list of norm violation counts per run.
  - The final accumulated hypothesis inferred using ACUITY.

---

## Algorithm Overview

Each iteration of the main loop performs the following steps:

1. Update the reinforcement learning (RL) policy.
2. Infer a seed based on the learned behavior.
3. Generalize the seed with **ACUITY**.
4. Deduce all state-action pairs covered by the new hypothesis.
5. Add these pairs to the set of invalid state-actions.
6. Count the number of **norm violations** in the test trajectories.

---

## Project Structure

```
nsicrl/
├── src/               # Source code and main logic
├── requirements.txt   # Python dependencies
├── README.md          # This file
└── ...
```

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 📣 Citation

If you use this code, please cite the paper:

```
@inproceedings{YourCitationKey,
  title     = {Neuro-Symbolic Inverse Constrained Reinforcement Learning},
  author    = {Deane, Oliver and Ray, Oliver},
  year      = {in press},
  booktitle = {International Conference on Neuro-symbolic AI},
  ...
}
```

---

## 📌 Notes

- Figures are generated automatically in `figures/`.
- All arguments are defaulted for convenience.

