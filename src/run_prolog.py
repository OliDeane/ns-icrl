from swiplserver import PrologMQI, PrologThread

def convert_integers_to_string(clause):
    new_clause = []
    for predicate in clause:
        predicate_args = [str(arg) for arg in predicate["args"]]
        new_clause.append({"functor": predicate["functor"], "args": predicate_args})
    return new_clause

def pop_the_head(predicate_list, label, head_predicate_string='active'):
    head_predicate = []
    new_predicate_list = []

    for predicate in predicate_list:
        if predicate.startswith(f'{head_predicate_string}('):
            head_string = predicate
        else:
            new_predicate_list.append(predicate)

    head_string = 'None'
    body_string = ', '.join(new_predicate_list)

    if label == 'neg':
        head_string = head_string.replace(head_predicate, 'negative_class')

    return head_string, body_string

def run_acuity():
    
    pycuity_path = "prolog_files/acuityFiles/pycuity.pl"
    acuity_data_path = "'prolog_files/sumo_bidir_4way_junction2/sumo_bidir_4way_junction2'"
    with PrologMQI() as mqi:
        with mqi.create_thread() as main_prolog_thread:
            main_prolog_thread.query(f"['{pycuity_path}'].")
            main_prolog_thread.query(f"read_all({acuity_data_path}).")

            main_prolog_thread.query("induce.")

            # Run the positive prediction inference on the main thread
            hyp = main_prolog_thread.query("get_hypothesis_as_list(HypothesisBody).")

            hypothesis_body = hyp[0]['HypothesisBody']    

            label = "test"
            pos_rules = []
            for clause in hypothesis_body:
                clause = convert_integers_to_string(clause)
                clause_as_list = [f"{item['functor']}({','.join(item['args'])})" for item in clause]
                head_string, body_string = pop_the_head(clause_as_list, label, head_predicate_string = "invalid")
                pos_rules.append("norm_violation(A)" + ' :- ' + body_string)

            # Save logical deductions of the hypothesis
            main_prolog_thread.query("save_state_actions_to_file.")

    return pos_rules
